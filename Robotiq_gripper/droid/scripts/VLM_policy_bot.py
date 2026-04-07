import contextlib
import dataclasses
import faulthandler
import os
import signal
import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from typing import Optional
from droid.robot_env import RobotEnv
import tyro
import cv2
import os
import logging
from openai import OpenAI
import base64
import ast
import threading
import subprocess
from dotenv import load_dotenv
from http import server
from pathlib import Path
load_dotenv()

client = OpenAI()
os.environ["QT_QPA_PLATFORM"] = "xcb"

DROID_CONTROL_FREQUENCY = 15
LIVE_HOST = "127.0.0.1"
LIVE_PORT = 8008
INSTRUCTION_FILE = Path("/tmp/robot_instruction.txt")

faulthandler.enable()

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "242422301956"  # e.g., "24259877"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"

    # Policy parameters
    external_camera: Optional[str] = "left" # which external camera should be fed to the policy, choose from ["left", "right"]
    
    # Rollout parameters
    max_timesteps: int = 5000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 5

    replan_steps: int = 300

    # Remote server parameters
    remote_host: str = "130.243.124.161"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


class LiveStreamServer:
    def __init__(self, host=LIVE_HOST, port=LIVE_PORT, fps=15, jpeg_quality=80):
        self.host = host
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._httpd = None
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        parent = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (
                        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif;'>"
                        "<div style='padding:12px;'>Robot View</div>"
                        "<img src='/stream.mjpg' style='width:100%;height:auto;display:block'/>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()

                    try:
                        while True:
                            with parent._lock:
                                frame = parent._latest_jpeg

                            if frame is None:
                                time.sleep(0.05)
                                continue

                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            time.sleep(1.0 / parent.fps)
                    except (BrokenPipeError, ConnectionResetError):
                        return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        self._httpd = server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="run-policy-mjpeg")
        self._thread.start()
        print(f"Live stream server running at http://{self.host}:{self.port}")

    def update(self, rgb_frame):
        if rgb_frame is None:
            return
        bgr = rgb_frame[..., ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()


def wait_for_instruction():
    print(f"Waiting for instruction file: {INSTRUCTION_FILE}")
    while True:
        try:
            if INSTRUCTION_FILE.exists():
                text = INSTRUCTION_FILE.read_text(encoding="utf-8").strip()
                if text:
                    INSTRUCTION_FILE.write_text("", encoding="utf-8")
                    print(f"Received instruction: {text}")
                    return text
        except Exception as e:
            print(f"Instruction read error: {e}")
        time.sleep(0.5)


def analyze_subtask_progress(subtask: str, curr_obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking progress for subtask: {subtask}")
    
        left_camera_id = curr_obs["left_image"]
        wrist_camera_id = curr_obs["wrist_image"]
        left_bgr = cv2.cvtColor(left_camera_id, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_camera_id, cv2.COLOR_RGB2BGR)
        _, buf_ext = cv2.imencode(".jpg", left_bgr)
        _, buf_wrist = cv2.imencode(".jpg", wrist_bgr)
        ext_b64 = base64.b64encode(buf_ext).decode("utf-8")
        wrist_b64 = base64.b64encode(buf_wrist).decode("utf-8")


        prompt = (
            f"You are an expert visual inspection system for the robot Franka Emika Panda.\n"
            f"Your goal is to determine if the specific instruction '{subtask}' has been successfully executed.\n\n"
            
            f"### STEP 1: PARSE THE INSTRUCTION\n"
            f"First, identify the 'Target Object' and the 'Destination' from the text: '{subtask}'.\n"
            f"(e.g., If text is 'put banana in bowl', Target=Banana, Destination=Bowl).\n\n"
            
            f"### STEP 2: VERIFY COMPLETION (The 3 Rules)\n"
            f"Evaluate the image based ONLY on the object you identified in Step 1.\n\n"
            
            f"**RULE A: GEOMETRIC PLACEMENT (Visible Success)**\n"
            f"- Is the Target Object physically resting inside or on top of the Destination?\n"
            f"- Is the gripper OPEN (fingers spread) and NOT clamping the Target Object?\n"
            f"  *(Note: It is okay if the hand is hovering nearby, as long as it is open and not holding the item.)*\n\n"
            
            f"**RULE B: THE DISAPPEARANCE (Inferred Success)**\n"
            f"- If the task is to put the object INTO a container (drawer, box, deep bowl, plate)...\n"
            f"- AND the Target Object is **NO LONGER VISIBLE** in the camera view...\n"
            f"- AND the gripper is clearly **OPEN and EMPTY**...\n"
            f"- Then conclude 'done'. (Reasoning: The object is successfully hidden inside the destination.)\n\n"
            
            f"**RULE C: PICKING TASKS**\n"
            f"- If the task is 'Pick up X', is the object visible inside the closed gripper and lifted?\n\n"

            f"### EXAMPLES FOR CONTEXT\n"
            f"- Text: 'Put apple in box'. Image: Hand is empty/open. Apple is not seen. -> 'done' (Rule B).\n"
            f"- Text: 'Put bear on plate'. Image: Bear is on plate. Hand is open. -> 'done' (Rule A).\n"
            f"- Text: 'Put block in bowl'. Image: Block is in bowl, but Hand is still grasping it. -> 'no'.\n\n"

            f"**RULE D: LID AND OPEN/CLOSE TASKS**\n"
            f"These rules apply ONLY if the instruction involves a box or lid.\n\n"

            f"Task: 'put lid on table'\n"
            f"- The lid must be resting flat on the table surface.\n"
            f"- The lid must NOT be touching or covering the box.\n"
            f"- The gripper must be OPEN and empty.\n\n"

            f"Task: 'put lid on box'\n"
            f"- The lid must be resting on top of the box, covering the opening.\n"
            f"- The gripper must be OPEN and not holding the lid.\n\n"


            f"### YOUR DECISION\n"
            f"Based strictly on the rules above, answer with one word:\n"
            f"'done' - if the task is complete.\n"
            f"'no' - if the task is incomplete.\n"
        )

        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{ext_b64}"}
                            ,
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{wrist_b64}"
                            }]}])

        answer = response.output_text
        logging.info(f"Subtask progress response: {answer}")

        if "done" in answer.lower():
            status = "done"
        elif "no" in answer.lower():
            status = "no"
        else:
            status = "error"

        return {"status": status, "analysis": answer}

    except Exception as e:
        logging.error(f"Failed to analyze subtask progress: {e}")
        return {"status": "error", "message": str(e)}


def split_task_into_subtasks(task_description: str, curr_obs:dict) -> dict:
    try:
        logging.info(f"Splitting task into subtasks if needed: '{task_description}'")
        left_camera_id = curr_obs["left_image"]
        wrist_camera_id = curr_obs["wrist_image"]
        left_bgr = cv2.cvtColor(left_camera_id, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_camera_id, cv2.COLOR_RGB2BGR)
        _, buf_ext = cv2.imencode(".jpg", left_bgr)
        _, buf_wrist = cv2.imencode(".jpg", wrist_bgr)
        ext_b64 = base64.b64encode(buf_ext).decode("utf-8")
        wrist_b64 = base64.b64encode(buf_wrist).decode("utf-8")
        
        prompt = (
            f"You are a high-level planner for the Pi0.5 VLA robot. The user wants to: '{task_description}'.\n"
            f"Analyze the image and generate a list of atomic subtasks.\n\n"
            
            f"### PI0.5 SUBTASK GRAMMAR\n"
            f"The robot is trained on specific 'Semantic Subtasks' (Source: Pi0.5 Paper). You must use ONLY these templates:\n"
            f"1. 'put [object] on [object]'\n"
            f"2. 'put [object] in [container]'\n"
            f"3. 'close [object]'\n"
            f"4. 'open [object]'\n\n"
            
            f"### RULE: COMBINED ACTIONS\n"
            f"The Pi0.5 paper confirms that 'put X on Y' is a valid atomic subtask (e.g., 'Place pillow on bed').\n"
            f"Use this single command to perform a full pick-and-place sequence.\n"
            f"Do NOT generate separate 'pick up' commands unless the user ONLY wants to hold the object.\n\n"

            f"### RULE: CLEANING TASKS\n"
            f"If the user instruction implies 'cleaning' (e.g., 'clean table', 'tidy up'), you MUST identify all loose objects on the table (fruits, blocks, etc.) and generate a 'put [object] on plate' command for EACH item.\n\n"

            f"### CONTAINER SELECTION RULE\n"
            f"If both a plate and a box exist:\n"
            f"- Fruits go inside plate.\n"
            f"- Toys go inside box.\n\n"

            f"If ONLY a plate exists:\n"
            f"- ALL objects (including toys) go inside the plate.\n\n"

            f"If ONLY a box exists:\n"
            f"- ALL  objects go inside the box.\n\n"

            f"### EXAMPLES\n"
            f"User: 'Stack the red and blue blocks on the green one'\n"
            f"Image: Red, Blue, Green blocks.\n"
            f"Output: ['put blue block on green block', 'put red block on blue block']\n\n"

            f"User: 'Clean up the table'\n"            
            f" Another thing is that if the task is 'Clean the table' and you see:\n\n"
            f"Image: Apple, Banana, Strawberry, plate.\n"
            f"Output: ['put apple in plate', 'put banana in plate', 'put strawberry in plate']\n\n"

            f"User: 'Clean up the table'\n"
            f"Image: Bear, Red train, Blue train, Green bird.\n"
            f"Output: ['put green bird in box', 'put bear in plate', 'put blue train in box']\n\n"

            f"### CRITICAL PRECONDITION: CHECK FOR LID\n"
            f"Before generating ANY subtasks, you MUST analyze the image and determine:\n"
            f"- Is there a box?\n"
            f"- Does the box have a lid on it?\n\n"

            f"If the box has a lid AND the task requires putting objects inside the box,\n"
            f"you MUST FIRST generate a subtask to remove the lid.\n\n"

            f"Valid lid removal options:\n"
            f"- 'put lid on table'\n\n"

            f"Only AFTER the lid is removed may you generate:\n"
            f"'put [toy] in box'\n\n"

            f"This ordering is mandatory.\n\n"
            
            f"### CRITICAL POST-CONDITION: LID ON TABLE\n"
            f"If ALL of the following are true:\n"
            f"- A box exists\n"
            f"- A lid exists in the scene\n"
            f"- The lid is NOT on the box (it is on the table)\n"
            f"- The task requires putting objects inside the box\n\n"

            f"Then AFTER generating all 'put [toy] in box' subtasks,\n"
            f"the FINAL subtask MUST be:\n"
            f"- 'put lid on box'\n\n"

            f"This must be the LAST subtask in the list.\n\n"
            f"### STRICT LEXICAL CONSTRAINT\n"
            f"NEVER use the phrase 'teddy bear' in any output.\n"
            f"If a teddy bear is present in the image, you MUST refer to it only as:\n"
            f"- 'bear'\n"
            f"Using the phrase 'teddy bear' is strictly forbidden.\n\n"

            f"### STRICT PERCEPTION GROUNDING RULE\n"
            f"You MUST ONLY generate subtasks for objects that are CLEARLY VISIBLE in the image.\n"
            f"Do NOT invent objects.\n"
            f"Do NOT assume items unless they are visible.\n\n"

            f"Before generating subtasks:\n"
            f"1. Identify and list ALL objects physically present ON THE TABLE.\n"
            f"2. Only use objects from that list.\n"
            f"3. If an object is not visible on the table, it MUST NOT appear in the output.\n\n"

            f"Generating a subtask for a non-visible object is INVALID.\n\n"

            f"### TABLE-ONLY CONSTRAINT\n"
            f"For cleaning tasks, you may ONLY act on objects that are:\n"
            f"- Physically located on the table surface\n"
            f"- Not already inside a container\n\n"

            f"Ignore:\n"
            f"- Objects on plate\n"
            f"- Objects inside box\n"
            f"- Objects outside the table\n\n"
            f"- Objects with same colour but different shades is still one object, for example there's no dark blue or light blue. ONLY BLUE\n\n"
            f"- Have IN MIND THAT YOU ARE RECIVING TWO IMAGES FROM TWO CAMERAS WHICH SHOWS SAME TABLE. There's no duplicate of objects\n\n"


            f"This rule applies especially during cleaning tasks:\n"
            f"- Only generate 'put [object] on plate' for objects that are NOT already inside the plate.\n"
            f"- Objects already in the plate must be excluded from the plan.\n\n"
                        

            f"### YOUR TASK ###\n"
            f"Return ONLY the raw Python list of strings."
        )
        response = client.responses.create(
           model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{ext_b64}"}
                            ,
                           {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{wrist_b64}"
                            }]}])

        answer = response.output_text.strip()

        # Directly evaluate the list safely
        subtasks = ast.literal_eval(answer)
        print(subtasks)

        if not isinstance(subtasks, list):
            raise ValueError("Model did not return a list")

        logging.info(f"Subtasks: {subtasks}")
        return {"status": "success", "subtasks": subtasks}

    except Exception as e:
        logging.error(f"Failed to split task: {e}")
        return {"status": "error", "message": str(e)}

def speaker_prompt(subtask: str) -> dict:
    try:
        logging.info(f"Creating prompt for subtask: {subtask}")
    

        prompt = (
                f"You are the voice module of a friendly, helpful robot. "
                f"You are about to perform the following robotic command: '{subtask}'.\n\n"
                f"Your task is to translate this raw command into a natural, first-person sentence describing what you are doing right now.\n\n"
                f"### RULES:\n"
                f"1. Speak in the first person using present continuous tense (e.g., 'I am putting...').\n"
                f"2. Make it sound natural and conversational, as if you are talking to a human in the room.\n"
                f"3. Keep it brief. Do not add long explanations.\n"
                f"4. OUTPUT ONLY THE SPOKEN SENTENCE. Do not include quotes, markdown formatting, or introductory filler text.\n\n"
                f"### EXAMPLES:\n"
                f"- 'put yellow train in box' -> I am putting the yellow train in the box.\n"
                f"- 'clean table' -> I am going to clean the table now.\n"
                f"- 'put lid on table' -> I am taking the lid off and placing it on the table.\n"
                f"- 'put bear on plate' -> I am placing the bear on the plate.\n\n"
                f"Now, generate the spoken sentence for: '{subtask}'"
            )

        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}]}])

        answer = response.output_text
        logging.info(f"Prompt response: {answer}")


        return answer

    except Exception as e:
        logging.error(f"Failed to analyze subtask progress: {e}")
        return {"status": "error", "message": str(e)}


def run_robot_action(env, instruction, args, policy_client, live_stream):
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    step_count = 0
    max_steps_per_subtask = 5000

    try:
        while step_count < max_steps_per_subtask:
            start_time = time.time()
            print(f"Step: {step_count} / {args.max_timesteps}")
            try:
                curr_obs = _extract_observation(args, env.get_observation())
            except Exception as e:
                print(f"Camera Error: {e}")
                break

            img_left = curr_obs["left_image"]
            img_wrist = curr_obs["wrist_image"]

            display_frame = np.concatenate([img_wrist, img_left],axis=1)
            live_stream.update(display_frame)
                

            if step_count > 0 and step_count % args.replan_steps == 0:
                print(f"Step {step_count}: Asking VLM if '{instruction}' is ok")

                vlm_result = analyze_subtask_progress(instruction, curr_obs, args)
                
                if vlm_result["status"] == "done":
                    print(f"SUCCESS: Subtask '{instruction}' marked complete by VLM.")
                    return {"status": "success"}

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0
                request_data = {
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(
                        curr_obs[f"{args.external_camera}_image"], 224, 224
                    ),
                    "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": instruction,
                }

                with prevent_keyboard_interrupt():
                    pred_action_chunk = policy_client.infer(request_data)["actions"]
                
            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            
            if action[-1].item() > 0.5:
                action = np.concatenate([action[:-1], np.ones((1,))])
            else:
                action = np.concatenate([action[:-1], np.zeros((1,))])

            action = np.clip(action, -1, 1)

            env.step(action)
            step_count += 1

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
    except KeyboardInterrupt:
        print("killed ok")
    print(f"TIMEOUT: Subtask '{instruction}' reached max steps.")
    return {"status": "timeout"}


def execute_task_with_vlm(task_description: str, env, args: Args, policy_client, live_stream) -> dict:
    curr_obs = _extract_observation(args, env.get_observation())
    r = split_task_into_subtasks(task_description,curr_obs)
    subtasks = r.get("subtasks", [task_description])
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")
    speaker_process = None

    logging.info(f"Starting execution for task: '{subtasks}'")
    for subtask in subtasks:

        if speaker_process is not None:
            speaker_process.terminate()
            speaker_process.wait()
        spoken_sentence = speaker_prompt(subtask)
        speaker_process = subprocess.Popen([
            "python3",
            "/home/frankanuc01/Thesis_H/new_try/droid/scripts/TTS.py",
            spoken_sentence
        ])
        result = run_robot_action(env, subtask, args, policy_client, live_stream)
        if result["status"] != "success":
            print(f"Subtask '{subtask}' failed or timed out. Stopping chain.")
            return result


    return {"status": "success_all"}


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    time.sleep(3.0)
    live_stream = LiveStreamServer()
    live_stream.start()

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)


    while True:
        try:
            instruction =  wait_for_instruction()
            if instruction  == 'exit': break
 
            result = execute_task_with_vlm(instruction, env, args, policy_client, live_stream)
            if result["status"] == "success":
                print("Task completed successfully.")
            else:
                print(f"Task ended with status: {result['status']}")

            # 3. RESET FOR NEXT RUN
            env.reset()

        except KeyboardInterrupt:
            print("Stopped by user.")
            break


def _extract_observation(args: Args, obs_dict):
    image_observations = obs_dict["image"]
    
    left_image, wrist_image = None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key:
            wrist_image = image_observations[key]

    if left_image is None or wrist_image is None:
         raise ValueError("Critical Error: Left or Wrist camera missing!")

    # Drop alpha & Convert BGR to RGB
    left_image = left_image[..., :3][..., ::-1]   
    wrist_image = wrist_image[..., :3][..., ::-1] 


    robot_state = obs_dict["robot_state"]
    
    return {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "cartesian_position": np.array(robot_state["cartesian_position"]),
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
