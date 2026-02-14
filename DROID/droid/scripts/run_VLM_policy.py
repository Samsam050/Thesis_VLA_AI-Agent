import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from typing import Optional, Union 
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import cv2
import os
import logging
from openai import OpenAI
import base64
import ast
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
os.environ["QT_QPA_PLATFORM"] = "xcb"

DROID_CONTROL_FREQUENCY = 15
#prompt i used:
# ok sucess: Pick up the yellow banana **from the table** and put it in the pink bowl.  
# bad sucess: Pick up the yellow banana **from the table** and drop it in the pink bowl. Pick up the yellow banana from the table and **place it in the pink bowl and let go**.

faulthandler.enable()

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "242422301956"  # e.g., "24259877"
    #left_camera_id: str = "949122070317"
    right_camera_id: str = ""  # e.g., "24514023"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"

    # Policy parameters
    external_camera: Optional[str] = "left" # which external camera should be fed to the policy, choose from ["left", "right"]
    

    # Rollout parameters
    max_timesteps: int = 2000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 5

    replan_steps: int = 150
    #for each 250 we replan

    # Remote server parameters
    remote_host: str = "130.243.124.173"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

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
            f"- If the task is to put the object INTO a container (drawer, box, deep bowl)...\n"
            f"- AND the Target Object is **NO LONGER VISIBLE** in the camera view...\n"
            f"- AND the gripper is clearly **OPEN and EMPTY**...\n"
            f"- Then conclude 'done'. (Reasoning: The object is successfully hidden inside the destination.)\n\n"
            
            f"**RULE C: PICKING TASKS**\n"
            f"- If the task is 'Pick up X', is the object visible inside the closed gripper and lifted?\n\n"

            f"### EXAMPLES FOR CONTEXT\n"
            f"- Text: 'Put apple in box'. Image: Hand is empty/open. Apple is not seen. -> 'done' (Rule B).\n"
            f"- Text: 'Put bear on plate'. Image: Bear is on plate. Hand is open. -> 'done' (Rule A).\n"
            f"- Text: 'Put block in bowl'. Image: Block is in bowl, but Hand is still grasping it. -> 'no'.\n\n"

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
            f"If the user instruction implies 'cleaning' (e.g., 'clean table', 'tidy up'), you MUST identify all loose objects on the table (fruits, blocks, cans, etc.) and generate a 'put [object] in bowl' command for EACH item.\n\n"

            f"### EXAMPLES\n"
            f"User: 'Stack the red and blue blocks on the green one'\n"
            f"Image: Red, Blue, Green blocks.\n"
            f"Output: ['put blue block on green block', 'put red block on blue block']\n\n"

            f"User: 'Clean up the table'\n"
            f"Image: Apple, Banana, Sponge, Bowl.\n"
            f"Output: ['put apple in bowl', 'put banana in bowl', 'put sponge in bowl']\n\n"

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

def run_robot_action(env, instruction, args, policy_client, display):
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    step_count = 0
    max_steps_per_subtask = 2000
    
    current_gripper_state = 1.0
    try:
        while step_count < max_steps_per_subtask:
            start_time = time.time()
            print(f"Step: {step_count} / {args.max_timesteps}")
            try:
                curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=False)
            except Exception as e:
                print(f"Camera Error: {e}")
                break

            img_left = curr_obs["left_image"]
            img_wrist = curr_obs["wrist_image"]
            h_target = 480
            w_left = int(img_left.shape[1] * (h_target / img_left.shape[0]))
            w_wrist = int(img_wrist.shape[1] * (h_target / img_wrist.shape[0]))
            disp_left = cv2.resize(img_left, (w_left, h_target))
            disp_wrist = cv2.resize(img_wrist, (w_wrist, h_target))
            
            display_frame = np.concatenate([disp_left, disp_wrist], axis=1)
            display.show(display_frame)
                

            if step_count > 0 and step_count % args.replan_steps == 0:
                print(f"Step {step_count}: Asking VLM if '{instruction}' is ok")

                vlm_result = analyze_subtask_progress(instruction, curr_obs, args)
                
                if vlm_result["status"] == "done":
                    print(f"SUCCESS: Subtask '{instruction}' marked complete by VLM.")
                    return {"status": "success"}

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0
                print("before sending:")
                print(curr_obs["gripper_position"])
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
            print("what get from model:")
            print(action[-1])
            # Binarize gripper
            future_average = np.mean(pred_action_chunk[-3:, -1])
            
            if future_average> 0.5 :
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


def execute_task_with_vlm(task_description: str, env, args: Args, policy_client, display) -> dict:
    curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=False)
    r = split_task_into_subtasks(task_description,curr_obs)
    subtasks = r.get("subtasks", [task_description])
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")

    logging.info(f"Starting execution for task: '{subtasks}'")
    for subtask in subtasks:
        result = run_robot_action(env, subtask, args, policy_client, display)
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

class Display:
    def __init__(self):
        cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)
        cv2.startWindowThread() 
    def show(self, frame):
        if frame is not None:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Robot View", bgr_frame)
            cv2.waitKey(1)
            cv2.pollKey()


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")
    time.sleep(3.0)
    display = Display()

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)


    while True:
        try:
            instruction = input("Enter instruction: ")
            if instruction  == 'exit': break
 
        
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            result = execute_task_with_vlm(instruction, env, args, policy_client, display)
            if result["status"] == "success":
                print("Task completed successfully.")
            else:
                print(f"Task ended with status: {result['status']}")

            # 3. RESET FOR NEXT RUN
            env.reset()

        except KeyboardInterrupt:
            print("Stopped by user.")
            break


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]
        # Only check right camera if ID exists
        elif args.right_camera_id and args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]

    if left_image is None or wrist_image is None:
         raise ValueError("Critical Error: Left or Wrist camera missing!")

    # Drop alpha & Convert BGR to RGB
    left_image = left_image[..., :3][..., ::-1]   
    wrist_image = wrist_image[..., :3][..., ::-1] 

    if right_image is not None:
        right_image = right_image[..., :3][..., ::-1]

    if save_to_disk:
        valid_imgs = [x for x in [left_image, wrist_image, right_image] if x is not None]
        
        # Find the smallest height (probably 480)
        min_h = min(img.shape[0] for img in valid_imgs)
        
        # Crop everything to that height (img[:480, ...])
        cropped_imgs = [img[:min_h] for img in valid_imgs]
        
        combined_image = np.concatenate(cropped_imgs, axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    robot_state = obs_dict["robot_state"]
    
    return {
        "left_image": left_image,
        "right_image": right_image, 
        "wrist_image": wrist_image,
        "cartesian_position": np.array(robot_state["cartesian_position"]),
        "joint_position": np.array(robot_state["joint_positions"]),
        "gripper_position": np.array([robot_state["gripper_position"]]),
    }

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
