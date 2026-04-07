import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
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
STEPS_LOG_FILE = "steps.txt"


faulthandler.enable()

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "242422301956"  # e.g., "24259877"
    #left_camera_id: str = "19892622"
    right_camera_id: str = "19443010C14F9D2E00"  # e.g., "24514023"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"
    #wrist_camera_id: str = "19892622"

    # Policy parameters
    external_camera: Optional[str] = "left" # which external camera should be fed to the policy, choose from ["left", "right"]
    

    # Rollout parameters
    max_timesteps: int = 5000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 4
    #open_loop_horizon: int = 8

    #replan_steps: int = 225
    replan_steps: int = 200
    #for each 250 we replan

    # Remote server parameters
    remote_host: str = "130.243.124.161"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


def initialize_steps_file(file_path: str = STEPS_LOG_FILE):
    """Create or clear the steps log file for a new task run."""
    with open(file_path, "w") as f:
        f.write("")


def append_completed_step(step_count: int, file_path: str = STEPS_LOG_FILE):
    """Append the completed step count to the file, separated by spaces."""
    with open(file_path, "a") as f:
        f.write(f"{step_count} ")


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
        f"You are a visual progress checker for a Franka robot.\n"
        f"Your job is to decide whether this ONE subtask is already complete:\n"
        f"'{subtask}'\n\n"

        f"You are shown TWO images of the SAME physical scene from TWO camera views.\n"
        f"Use both views together.\n"
        f"Do NOT count the same object twice.\n\n"

        f"Check ONLY the target object and the destination named in the instruction.\n"
        f"Ignore all other objects unless they help determine that the robot has moved on to a different task.\n"
        f"Ignore future planning.\n"
        f"Ignore whether the whole table is cleaned.\n"

        f"Important policy:\n"
        f"This checker is completion-biased.\n"
        f"If the target object is no longer visible in the scene and the robot is not holding it anymore, prefer 'done'.\n"
        f"Do NOT be overly conservative.\n"
        f"Do NOT require perfect proof.\n"
        f"Do NOT require the full destination to be visible for absence-based success.\n\n"

        f"This is a LOCAL check for ONE subtask only.\n\n"
        f"Task-specific override:\n"
        f"- For box-removal tasks and lid subtasks, use the special rules below even if they are stricter than the general rules.\n"
        f"- Visible evidence has higher priority than guessing from robot motion.\n\n"
        f"- For any subtask of the form 'put [object] on table', the checker must visually confirm the object on the table before answering 'done'.\n"
        f"- For table-placement subtasks, absence-based success is NOT allowed.\n"
        f"- For table-placement subtasks, moved-on success is NOT allowed.\n\n"

        f"Decision rules in PRIORITY ORDER:\n\n"

        f"1. DIRECT SUCCESS\n"
        f"- Answer 'done' if the target object is clearly visible at the destination.\n"
        f"- For 'in' tasks, answer 'done' if the target object is clearly inside the named container.\n"
        f"- For 'on' tasks, answer 'done' if the target object is clearly on the named destination.\n"
        f"- For 'put object on table' or 'put [object] on table', answer 'done' if the target object is clearly resting on the table,\n"
        f"  not inside the box, not inside another container, and not in the gripper.\n\n"
        f"- For table-placement subtasks, direct visual confirmation on the table is REQUIRED for 'done'.\n\n"

        f"2. ABSENCE-BASED SUCCESS\n"
        f"- Answer 'done' if the target object is not visible in either image,\n"
        f"- and there is no evidence that the gripper is still holding the target object,\n"
        f"- and there is no evidence that the robot is still actively placing that same target object.\n"
        f"- This rule is strong, but it must NOT override clear visible evidence that the target object is still away from the destination.\n"
        f"- If the target object has disappeared from both views after being manipulated, treat that as completed.\n\n"
        f"- IMPORTANT: Do NOT use absence-based success for subtasks whose destination is the table.\n"
        f"- If the subtask is 'put [object] on table' and the object is not clearly visible on the table, answer 'no'.\n\n"

        f"3. MOVED-ON SUCCESS\n"
        f"- Answer 'done' if the robot is clearly doing a different task now ONLY when there is no visible evidence that this subtask is still unfinished.\n"
        f"- Do NOT use moved-on behavior to mark lid subtasks as done when the lid is still visibly in the wrong place.\n"
        f"- Examples:\n"
        f"  a) the gripper is empty and away from the old target,\n"
        f"  b) the gripper is interacting with a different object,\n"
        f"  c) the robot posture and scene indicate the old target is no longer being manipulated,\n"
        f"  d) the target object is not visible anymore and the robot has already shifted attention elsewhere.\n"
        f"- Do NOT answer 'done' from moved-on behavior if the target object is still clearly visible away from the destination.\n"
        f"- Visible unfinished evidence beats moved-on guesses.\n\n"
        f"- IMPORTANT: Do NOT use moved-on success for subtasks whose destination is the table.\n"
        f"- For 'put [object] on table', the robot doing something else does NOT mean success.\n\n"

        f"4. OCCLUDED CONTAINER SUCCESS\n"
        f"- For container tasks like 'put green cube in box' or 'put banana in bowl', answer 'done' if the target object is hidden by the container,\n"
        f"- as long as the target object is not visible anywhere else in either image,\n"
        f"- and the robot is not still holding or actively placing it.\n"
        f"- Do NOT require seeing the entire object once it has dropped into the container.\n\n"

        f"5. STILL IN PROGRESS OR FAILURE\n"
        f"- Answer 'no' if the gripper is clearly holding the target object.\n"
        f"- Answer 'no' if the target object is clearly visible away from the destination.\n"
        f"- Answer 'no' if the robot is clearly in the middle of placing the target object right now.\n"
        f"- Answer 'no' if the target object is still visible near the gripper and has not been released.\n\n"
        f"- For 'put [object] on table', answer 'no' if the object is not clearly visible on the table.\n"
        f"- For 'put [object] on table', answer 'no' if the object is still visible inside the box or another container.\n"
        f"- For 'put [object] on table', answer 'no' if the object is not visible at all.\n\n"

        f"6. OCCLUSION CAUTION\n"
        f"- Answer 'no' only when the most likely explanation is that the target object is merely hidden by the robot arm or gripper during the same placement attempt.\n"
        f"- But if the target object is absent from both views and the robot has already moved on, answer 'done', not 'no'.\n\n"

        f"Box-removal rules:\n"
        f"- For subtasks like 'put object on table' or 'put red cube on table' that come from taking something out of a box,\n"
        f"  answer 'done' ONLY if the target object is clearly visible on the table and is no longer inside the box or in the gripper.\n"
        f"- Direct visual confirmation on the table is REQUIRED.\n"
        f"- If the target object is still clearly inside the box, answer 'no'.\n"
        f"- If the target object is still clearly in the gripper above or near the box, answer 'no'.\n"
        f"- If the target object is not visible in either image, answer 'no'.\n"
        f"- Do NOT answer 'done' just because the robot moved away.\n"
        f"- Do NOT answer 'done' just because the object is not visible.\n"
        f"- Do NOT infer table placement from absence.\n\n"

        f"Lid rules:\n"
        f"- For 'put lid on table': answer 'done' only if the lid is on the table and not covering the box.\n"
        f"- For 'put lid on table': answer 'no' if the lid is still covering the box or still in the gripper.\n"
        f"- For 'put lid on box': answer 'done' only if the lid is covering the box.\n"
        f"- For 'put lid on box': answer 'no' if the lid is still on the table or still in the gripper.\n\n"
        f"- Lid subtasks are STRICT.\n"
        f"- For lid subtasks, absence-based success and moved-on success must NOT override visible evidence.\n"
        f"- If the lid is visible and in the wrong place, answer 'no'.\n\n"

        f"- For lid subtasks, visible lid position is decisive.\n"
        f"- If the lid is still visible on the box, return 'no' for 'put lid on table'.\n"
        f"- If the lid is still visible on the table, return 'no' for 'put lid on box'.\n"
        f"- Do not use absence-based success or moved-on success for lid subtasks when the lid is still visible.\n\n"

        f"Examples:\n"
        f"- Subtask: 'put green cube in box'\n"
        f"- If the green cube is visible in the box: done.\n"
        f"- If the green cube is not visible in either image, and the gripper is not holding it: done.\n"
        f"- If the green cube is not visible, and the robot is already handling another object: done.\n"
        f"- If the green cube is still visible on the table or in the gripper: no.\n"
        f"- If the green cube is only temporarily hidden by the arm during the same placement motion: no.\n\n"
        f"- Subtask: 'put lid on table'\n"
        f"- If the lid is clearly resting on the table and the box is uncovered: done.\n"
        f"- If the lid is still on the box or still in the gripper: no.\n\n"
        f"- Subtask: 'put object on table'\n"
        f"- If the object taken from the box is clearly on the table and not in the gripper: done.\n"
        f"- If the object is still inside the box or still being lifted out: no.\n\n"
        f"- Subtask: 'put bear on table'\n"
        f"- If the bear is clearly visible resting on the table: done.\n"
        f"- If the bear is still inside the box: no.\n"
        f"- If the bear is not visible in either image: no.\n"
        f"- If the robot has moved away but the bear is not clearly visible on the table: no.\n\n"

        f"When the target object has disappeared from both views and the robot is no longer engaged with it, answer 'done' ONLY if no stricter rule above says 'no'.\n"
        f"Final instruction:\n"
        f"For lid subtasks, visible lid position has higher priority than robot motion or guessing.\n"
        f"If the lid is still visibly on the box, answer 'no'.\n"
        f"If the lid is still visibly on the table, answer 'no' for 'put lid on box'.\n\n"
        f"For any subtask whose destination is the table, answer 'done' ONLY if the target object is clearly visible on the table.\n"
        f"If the target object is not clearly visible on the table, answer 'no'.\n"
        f"Do NOT use absence-based success or moved-on success for table-placement subtasks.\n\n"

        f"Answer with ONLY ONE WORD:\n"
        f"- Return exactly 'done' if the subtask is complete.\n"
        f"- Return exactly 'no' if the subtask is not complete.\n"
        f"- If you are uncertain, return exactly 'no'.\n"
        f"- Do not add punctuation.\n"
        f"- Do not add explanation.\n"
        f"- Do not write 'not done'.\n"
        f"- Output must be exactly one of: done or no.\n"
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
            f"You are a strict high-level planner for the Pi0.5 VLA robot.\n"
            f"The user wants to: '{task_description}'.\n\n"

            f"You are shown TWO images of the SAME table scene from TWO camera views.\n"
            f"They show the SAME physical objects from different angles.\n"
            f"The two images are NOT two separate scenes.\n"
            f"An object that appears in both images is usually the SAME physical object.\n\n"

            f"First, mentally match objects across the two views.\n"
            f"Merge any objects that are the same physical item.\n"
            f"Then plan using the set of UNIQUE physical objects only.\n"
            f"Never count the same physical object twice just because it appears in both views.\n"
            f"If a blue cube appears in both images, and there is no clear evidence of two different blue cubes, treat it as ONE blue cube.\n\n"

            f"You must generate a Python list of atomic subtasks using ONLY this form:\n"
            f"1. 'put [object] on [object]'\n"
            f"2. 'put [object] in [container]'\n\n"
            f"Priority rule:\n"
            f"- Task-specific rules below OVERRIDE the general rules when they conflict.\n\n"
            f"Rules:\n"
            f"- Use ONLY objects that are clearly visible in at least one view.\n"
            f"- Do NOT invent objects.\n"
            f"- Count UNIQUE physical objects, not image appearances.\n"
            f"- Only create subtasks for loose objects on the table, EXCEPT for explicit box-removal tasks described below.\n"
            f"- Ignore objects already inside a box, EXCEPT for explicit box-removal tasks described below.\n"
            f"- Ignore objects already inside a bowl.\n"
            f"- Ignore objects outside the table.\n"
            f"- Use simple object names.\n"
            f"- Use 'bear', never 'teddy bear'.\n"
            f"- Treat colour as one colour only: use 'blue', not 'light blue' or 'dark blue'.\n"
            f"- When referring to a bowl in any subtask, ALWAYS include its colour (e.g., 'blue bowl', 'red bowl').\n"
            f"- Each loose object may be moved at most ONCE in the plan.\n"
            f"- Do NOT create two subtasks for the same physical object.\n"
            f"- If uncertain whether two same-looking objects in different views are one object or two, prefer treating them as ONE object unless both are clearly visible at the same time in one view or there is strong evidence they are different objects.\n\n"

            f"Cleaning rules:\n"
            f"- If both a bowl and a box are visible: fruits go in bowl, toys go in box.\n"
            f"- If only a bowl is visible: all loose objects go in bowl.\n"
            f"- If only a box is visible: all loose objects go in box.\n\n"

            f"Lid rules:\n"
            f"- Only use lid actions if a lid is clearly visible.\n"
            f"- If a box is clearly covered by a lid and objects must go in the box, first output: 'put lid on table'\n"
            f"- After all objects are in the box, if the lid is visible on the table, final output may be: 'put lid on box'\n\n"

            f"Box-removal rules:\n"
            f"- If the user asks to take out, remove, or retrieve an object from a box, this is a special box-removal task.\n"
            f"- For a box-removal task, objects inside the box are RELEVANT and must NOT be ignored.\n"
            f"- If the box is covered by a visible lid, the FIRST subtask must be: 'put lid on table'\n"
            f"- Then output a subtask that places the target object from the box onto the table.\n"
            f"- If the target object inside the box is clearly visible, name it normally, e.g. 'put red cube on table'.\n"
            f"- If the task says 'the object' and the target object inside the closed box is not yet visible, you may use the generic name 'object' and output: 'put object on table'\n"
            f"- After the target object has been taken out, if the lid is visible on the table, the FINAL subtask must be: 'put lid on box'\n"
            f"- Do NOT move unrelated loose objects on the table during a box-removal task unless the user explicitly asked for that.\n"
            f"- Do NOT remove more than the requested target object(s) from the box.\n\n"

            f"Examples:\n"
            f"User: 'Clean up the table'\n"
            f"Image: pear, banana, strawberry, bowl\n"
            f"Output: ['put pear in bowl', 'put banana in bowl', 'put strawberry in bowl']\n\n"

            f"User: 'Take out the object from the box'\n"
            f"Image: box with lid on top\n"
            f"Output: ['put lid on table', 'put object on table', 'put lid on box']\n\n"

            f"User: 'Stack the red and blue blocks on the green one'\n"
            f"Image: red block, blue block, green block\n"
            f"Output: ['put blue block on green block', 'put red block on blue block']\n\n"

            f"If there's no object, you return an empty list of strings.\n"


            f"Return ONLY the raw Python list of strings.\n"
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

def verify_remaining_subtasks(task_description: str, env, args: Args) -> dict:
    try:
        curr_obs = _extract_observation(args, env.get_observation())
        result = split_task_into_subtasks(task_description, curr_obs)

        if result["status"] != "success":
            return result

        remaining = result["subtasks"]
        """IF theres no object on table left then return complete"""
        if len(remaining) == 0:
            logging.info("Task is completley done")
            return {"status": "complete", "remaining_subtasks": []}
        
        """Otherwise we return remaning subtask"""
        logging.info(f"Remaining subtasks found: {remaining}")
        return {"status": "incomplete", "remaining_subtasks": remaining}

    except Exception as e:
        logging.error(f"Failed to verify if there's remaing task: {e}")
        return {"status": "error", "message": str(e)}

def run_robot_action(env, instruction, args, policy_client, display):
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
            

            display.show("External View", img_left)
            display.show("Wrist View", img_wrist)
                                
            if step_count > 0 and step_count % args.replan_steps == 0:
                print(f"Step {step_count}: Asking VLM if '{instruction}' is ok")

                vlm_result = analyze_subtask_progress(instruction, curr_obs, args)
                
                if vlm_result["status"] == "done":
                    print(f"SUCCESS: Subtask '{instruction}' marked complete by VLM.")
                    return {"status": "success", "completed_step":step_count}

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0

                print(instruction)
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
            #future_average = np.mean(pred_action_chunk[-3:, -1])
            
            #if future_average> 0.6 :
            if action[-1].item() > 0.5:
                action = np.concatenate([action[:-1], np.ones((1,))])
                print("close")
            else:
                action = np.concatenate([action[:-1], np.zeros((1,))])
                print("open")

            action = np.clip(action, -1, 1)

            env.step(action)
            step_count += 1

            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
    except KeyboardInterrupt:
        print("killed ok")
    print(f"TIMEOUT: Subtask '{instruction}' reached max steps.")
    return {"status": "timeout", "completed_step": step_count}


def execute_task_with_vlm(task_description: str, env, args: Args, policy_client, display) -> dict:
    initialize_steps_file()
    curr_obs = _extract_observation(args, env.get_observation())
    r = split_task_into_subtasks(task_description,curr_obs)

    subtasks = r.get("subtasks", [task_description])
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")

    logging.info(f"Starting execution for task: '{subtasks}'")
    for subtask in subtasks:
        result = run_robot_action(env, subtask, args, policy_client, display)
        if result["status"] != "success":
            print(f"Subtask '{subtask}' failed or timed out. Stopping chain.")
            return result
        
        completed_step = result.get("completed_step")
        if completed_step is not None:
            append_completed_step(completed_step)
            print(f"Wrote completed step {completed_step} to {STEPS_LOG_FILE}")

        #rest so it goes back to inital pos
        print("resetting")
        time.sleep(1.0)
        env.reset()
        time.sleep(1.0)
    
    verification = verify_remaining_subtasks(task_description, env, args)

    if verification["status"] == "complete":
        return {"status": "success_all"}

    if verification["status"] == "incomplete":
        remaining_subtasks = verification["remaining_subtasks"]

        for subtask in remaining_subtasks:
            result = run_robot_action(env, subtask, args, policy_client, display)
            if result["status"] != "success":
                print(f"Subtask '{subtask}' failed or timed out. Stopping chain.")
                return result
            
            completed_step = result.get("completed_step")
            if completed_step is not None:
                append_completed_step(completed_step)
                print(f"Wrote completed step {completed_step} to {STEPS_LOG_FILE}")

            print("resetting")
            time.sleep(1.0)
            env.reset()
            time.sleep(1.0)
    
        return {"status": "success_all"}
    return verification


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
        cv2.namedWindow("External View", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist View", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("External View", 960, 540)
        cv2.resizeWindow("Wrist View", 960, 540)

    def show(self, window_name, frame):
        if frame is not None:
            bgr_frame = frame[..., ::-1]   # RGB -> BGR
            cv2.imshow(window_name, bgr_frame)
            cv2.waitKey(1)


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
 
            result = execute_task_with_vlm(instruction, env, args, policy_client, display)
            if result["status"] == "success_all":
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
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key][::-1, ::-1]  

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
