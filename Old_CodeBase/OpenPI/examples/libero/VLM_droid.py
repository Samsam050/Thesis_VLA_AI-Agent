import collections
import dataclasses
import logging
import math
import pathlib
import cv2
import imageio
import third_party.libero.libero.libero.benchmark as benchmark
from third_party.libero.libero.libero import get_libero_path
from third_party.libero.libero.libero.envs import SegmentationRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import base64
from openai import OpenAI
import ast
import time
client = OpenAI()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224  # resolution used to fit DROID

#Copied from main.py in OpenPI
@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "130.243.124.173"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 4

    # LIBERO environment-specific parameters
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )#check out libero_suite_task.py in benchmark folder
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    # Utils
    seed: int = 7  # Random Seed (for reproducibility)

#Defining the tools used

def _obs_to_imgs(obs, args):
    img  = np.ascontiguousarray(obs["frontview_image"][::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    side_img  = np.ascontiguousarray(obs["sideview_image"][::-1, ::-1])
    robot_img   = np.ascontiguousarray(obs["robot0_robotview_image"][::-1, ::-1])
    front_img   = np.ascontiguousarray(obs["agentview_image"][::-1])

    img = image_tools.convert_to_uint8(
        cv2.resize(img, (args.resize_size, args.resize_size))
    )
    wrist_img = image_tools.convert_to_uint8(
        cv2.resize(wrist_img, (args.resize_size, args.resize_size))
    )
    side_img = image_tools.convert_to_uint8(
       image_tools.resize_with_pad(side_img, args.resize_size, args.resize_size)
    )
    robot_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(robot_img, args.resize_size, args.resize_size)
    )
    front_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(front_img, args.resize_size, args.resize_size)
    )

    return img, wrist_img,side_img,robot_img,front_img
"""
def analyze_scene_for_task(task_description: str, obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking if task '{task_description}' is possible...")

        img, wrist_img, side_img, robot_img,front_img = _obs_to_imgs(obs, args)
    
        _,buffer = cv2.imencode(".jpg", img)
        _,buffer2 = cv2.imencode(".jpg", wrist_img)
        _,buffer3 = cv2.imencode(".jpg", side_img)
        _,buffer4 = cv2.imencode(".jpg", robot_img)
        _,buffer5 = cv2.imencode(".jpg", front_img)
        img_base64 = base64.b64edncode(buffer).decode("utf-8")
        wrist_img_base64 = base64.b64encode(buffer2).decode("utf-8")
        side_img_base64 = base64.b64encode(buffer3).decode("utf-8")
        robot_img_base64 = base64.b64encode(buffer4).decode("utf-8")
        front_img_base64 = base64.b64encode(buffer5).decode("utf-8")

        prompt = (
            f"You are a robot helper. Look at the image. "
            f"Can the robot perform this task objectwise: '{task_description}' based on the image? You should not think about something else "
            f"Answer YES or NO. If NO, briefly explain why (e.g., object does not exist)."
        )

        # Call the OpenAI model
        response = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=[{
                    "role": "user", 
                    "content": [
                        {"type": "input_text", "text": prompt},
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{img_base64}"}
                            ,
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{wrist_img_base64}"
                            },
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{side_img_base64}"
                            },
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{robot_img_base64}"
                            },
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{front_img_base64}"
                            }        
                            ]
                            }]
        )

        answer = response.output_text
        logging.info(f"VLM response: {answer}")

        return {"status": "success", "analysis": answer}

    except Exception as e:
        logging.error(f"Failed to analyze scene: {e}")
        return {"status": "error", "message": str(e)}

def split_task_into_subtasks(task_description: str) -> dict:
    try:
        logging.info(f"Splitting task into subtasks if needed: '{task_description}'")

        prompt = (
            f"You are a planning assistant for a robot. "
            f"Given this task: '{task_description}', decide if it needs to be split into smaller subtasks. "
            f"For example, if task is 'Pick up bowl and hold it' no need to split into smaller subtask. But if task is 'Pick up Bowl and then put it on the stove' then divide the task into 'pick up bowl and hold it' and 'put the bowl on the stove '. "
            f"If no, return a single-element list with the same task. "
            f"Do not explain — only return a valid Python list."
        )

        response = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt}]}],
        )

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


def analyze_subtask_progress(subtask: str, obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking progress for subtask: {subtask}")
        img, wrist_img, side_img, robot_img,front_img = _obs_to_imgs(obs, args)
    
        _,buffer = cv2.imencode(".jpg", img)
        _,buffer2 = cv2.imencode(".jpg", wrist_img)
        _,buffer3 = cv2.imencode(".jpg", side_img)
        _,buffer4 = cv2.imencode(".jpg", robot_img)
        _,buffer5 = cv2.imencode(".jpg", front_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        wrist_img_base64 = base64.b64encode(buffer2).decode("utf-8")
        side_img_base64 = base64.b64encode(buffer3).decode("utf-8")
        robot_img_base64 = base64.b64encode(buffer4).decode("utf-8")
        front_img_base64 = base64.b64encode(buffer5).decode("utf-8")


        prompt = (
                f"You are an expert visual inspection system for the robot Franka Emika Panda.\n"
                f"Your goal is to determine if the subtask '{subtask}' is COMPLETED based ONLY on the images.\n\n"
                f"Evaluate strictly by the following criteria:\n"
                f"1. PICK Tasks: The target object must be visibly held in the closed gripper AND clearly lifted off the supporting surface.\n"
                f"2. PLACE Tasks: The object must be visibly released, the gripper must be open and detached, AND the object must be resting on the target surface.\n"
                f"3. OPEN/CLOSE Tasks: The mechanism must be visibly in its final required state.\n\n"
                f"Answer with strictly one of the following options:\n"
                f"1. done - if the goal state is fully and unambiguously achieved.\n"
                f"2. no - if the goal state is not achieved.\n"
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
                            "image_url": f"data:image/jpeg;base64,{img_base64}"}
                            ,
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{wrist_img_base64}"
                            },
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{side_img_base64}"
                            },
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{robot_img_base64}"
                            },
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{front_img_base64}"
                            }      
                            ]
                }],
        )

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
"""
def run_robot_action(env, obs, prompt, args, Client_openpi):

    open_loop_horizon = 8  # Re-plan every 8 steps (Standard DROID setting)
    actions_from_chunk_completed = 0
    pred_action_chunk = None

    try:
        t = 0
        max_steps = 250
        vlm_c = 14
        C = 0
        while t < max_steps:
            img, wrist_img,side_img,robot_img,front_img = _obs_to_imgs(obs, args)
            cv2.imshow("env", img)
            cv2.imshow("wrist", wrist_img)
            cv2.imshow("side", side_img)
            cv2.imshow("roboy", robot_img)
            cv2.imshow("front", front_img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
                actions_from_chunk_completed = 0
                two_fingers = obs["robot0_gripper_qpos"]
                gripper_width = np.abs(two_fingers[0] - two_fingers[1])   
          
                element = {
                    "observation/exterior_image_1_left": img,
                    "observation/wrist_image_left": wrist_img,
                    "observation/joint_position":obs["robot0_joint_pos"],
                    "observation/gripper_position": np.array([gripper_width]),
                    "prompt": str(prompt),
                }

                pred_action_chunk = Client_openpi.infer(element)["actions"]
            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1 

            velocity_cmd = action[:-1] # First 7
            raw_gripper  = action[-1]  # Last 1
            
            if raw_gripper > 0.5:
                gripper_cmd = 1.0  # closed
            else:
                gripper_cmd = -1.0  # open

            final_action = np.concatenate([velocity_cmd, [gripper_cmd]])
            final_action = np.clip(final_action, -1, 1)
            obs,reward,done,info = env.step(final_action)
            t += 1
            """
            C +=1
            if C % vlm_c == 0:
                print("Checking status..")
                progress = analyze_subtask_progress(prompt, obs, args)
                if progress.get("status") == "done":
                    return {"status": "success","obs" :obs}
                elif progress.get("status") == "no":
                    print(t)
                    continue
                elif progress.get("status") == "error":
                    logging.warning(f"Subtask '{prompt}' failed in VLM check.")
                    return {"status": "error", "obs": obs}
            """   
        logging.warning(f"Reached Max_steps")
        return {"status": "incomplete", "obs": obs,}

    except Exception as e:
        logging.error(f"Error during run_robot_action: {e}")
        return {"status": "error", "message": str(e), "obs": obs}


def execute_task_with_vlm(task_description: str, env, obs, args: Args, Client_openpi) -> dict:
    """
    r = split_task_into_subtasks(task_description)
    subtasks = r.get("subtasks", [task_description])
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")

    logging.info(f"Starting execution for task: '{subtasks}'")
    feasibility = analyze_scene_for_task(task_description, obs, args)
    if feasibility.get("status") != "success":
        logging.error("Scene analysis failed.")
        return {"status": "not_feasible", "obs": obs} 
        
    if "NO" in feasibility.get("analysis", "").upper():
        return {"status": "not_feasible", "message": "Task not feasible", "obs": obs}

    subtask_results = []
    
    for subtask in subtasks:
        logging.info(f"Executing subtask: '{subtask}'")
        result = run_robot_action(env, obs, subtask, args,Client_openpi)
        obs = result.get("obs", obs)
        subtask_results.append(result)
    """
    logging.info(f"Executing subtask: '{task_description}'")
    result = run_robot_action(env, obs, task_description, args,Client_openpi)
    obs = result.get("obs", obs)
    return result

def eval_libero(args: Args) -> None:
    # Needed for the simulation, otherwise bullshit
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(0)    
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    obs = env.reset()
    robot = env.robots[0]
    ctrl = robot.controller

    print("Controller type:", type(ctrl))
    print("Control dim:", ctrl.control_dim)

    print("Controller attributes:")
    print([a for a in dir(ctrl) if not a.startswith("_")])



    # Here begins the real thing   
    client_OpenPi = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    while True:
        #custom_prompt = input("Enter your prompt (type exit to stop): ")
        custom_prompt = "Pick up bowl and put it on plate"
        logging.info(f"\nTask: {custom_prompt}")

        if custom_prompt.lower() == "exit":
            print("Exiting program...")
            return 
        result = execute_task_with_vlm(custom_prompt, env, obs, args, client_OpenPi)
        obs = result["obs"]


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution,"controller": "JOINT_VELOCITY" }
    env = SegmentationRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
