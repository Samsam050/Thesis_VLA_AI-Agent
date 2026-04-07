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
import ast
import time
import ollama
import json
client = ollama.Client()
MODEL_NAME = 'qwen3-vl:8b'  


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

#Copied from main.py in OpenPI
@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "130.243.124.173"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 4

    #This will not be needed when not running as simulation
    # LIBERO environment-specific parameters
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )#check out libero_suite_task.py in benchmark folder
    #This could be needed when running irl

    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    # Utils
    seed: int = 7  # Random Seed (for reproducibility)

#Defining tools used

#Tool for both creating the simulation and also taking in the camera feed
def _obs_to_imgs(obs, args):
    img       = np.ascontiguousarray(obs["frontview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    #side_img  = np.ascontiguousarray(obs["sideview_image"][::-1, ::-1])
    #robot_img   = np.ascontiguousarray(obs["robot0_robotview_image"][::-1, ::-1])
    #front_img   = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
    )
    """
    side_img = image_tools.convert_to_uint8(
       image_tools.resize_with_pad(side_img, args.resize_size, args.resize_size)
    )
    robot_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(robot_img, args.resize_size, args.resize_size)
    )
    front_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(front_img, args.resize_size, args.resize_size)
    )
    """
    _,buffer = cv2.imencode(".jpg", img)
    _,buffer2 = cv2.imencode(".jpg", wrist_img)
    """
    _,buffer3 = cv2.imencode(".jpg", side_img)
    _,buffer4 = cv2.imencode(".jpg", robot_img)
    _,buffer5 = cv2.imencode(".jpg", front_img)
    """
    #This is needed for VLM, endcoding to base64
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    wrist_img_base64 = base64.b64encode(buffer2).decode("utf-8")
    """
    side_img_base64 = base64.b64encode(buffer3).decode("utf-8")
    robot_img_base64 = base64.b64encode(buffer4).decode("utf-8")
    front_img_base64 = base64.b64encode(buffer5).decode("utf-8")
    """
    return img,wrist_img,img_base64, wrist_img_base64

#VLM tool for analying the scene, mostly used for checking if a object exists and if the robot can perform a specifc task. 
def analyze_scene_for_task(task_description: str, obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking if task '{task_description}' is possible...")

        _,_,img_base64, wrist_img_base64 = _obs_to_imgs(obs, args)
        
        #Prompt we are sending to the VLM, right now i see no issue with it. 
        prompt = (
            f"You are a robot helper. Look at the image. "
            f"Can the robot perform this task objectwise: '{task_description}' based on the image? You should not think about something else "
            f"Answer YES or NO. If NO, briefly explain why (e.g., object does not exist)."
        )
        #Sending all image data to VLM (Ollama)
        all_image_data = [
        img_base64,       
        wrist_img_base64, 
        """ 
        side_img_base64,   
        robot_img_base64,  
        front_img_base64   
        """
    ]

        response = client.generate(
            model=MODEL_NAME,
            prompt = prompt,
            images=all_image_data,
            #stream = True
        )
        # --- DEBUGGING BLOCK ---

        try:
            logging.info(f"FULL RAW RESPONSE: {response}") 
            
            if hasattr(response, 'finish_reason'):
                logging.info(f"Finish Reason: {response.finish_reason}")
        except:
            pass
        # --------------------------------


        answer = response.response
        logging.info(f"VLM response: {answer}")

        if "YES" in answer:
            status = "success"
        else:
            status = "no"

        return {"status": status, "analysis": answer}

    except Exception as e:
        logging.error(f"Failed to analyze scene: {e}")
        return {"status": "error", "message": str(e)}

# If a task has subtasks, we create a list of subtasks. To help the robot perform better
def split_task_into_subtasks(task_description: str) -> dict:
    try:
        logging.info(f"Splitting task into subtasks if needed: '{task_description}'")

        # I had some issues with this prompt, where sometimes it could create subtask where it shouldn't
        prompt = (
            f"You are a planning assistant for a robotic VLA policy."
            f"Your job is to parse the user instruction '{task_description}' into a list of executable subtasks.\n"
            f"CRITICAL RULES FOR SPLITTING:\n"
            f"1. **ATOMIC ACTIONS:** Commands like 'Put X on Y', 'Place X in Y', or 'Move X to Y' are ATOMIC."
            f"The robot policy can handle the pick-and-place sequence automatically. "
            f"DO NOT split 'Put bowl on plate' into 'Pick' and 'Place'. Keep it as one task.\n"
            f"2.COMPOUND COMMANDS:Only split the task if the user EXPLICITLY connects two distinct actions "
            f"using words like 'and', 'then', 'after', or ',' (e.g., 'Pick up bowl AND put it on shelf').\n"
            f"3.EXAMPLE:\n"
            f"- Input: 'Put the red bowl on the plate' -> Output: ['Put the red bowl on the plate']\n"
            f"- Input: 'Pick up the bowl and put it on the shelf' -> Output: ['Pick up the bowl', 'Put the bowl on the shelf']\n"
            f"- Input: 'Open the drawer and put the apple inside' -> Output: ['Open the drawer', 'Put the apple inside the drawer']\n"
            f"Return ONLY a valid Python list of strings."
        )

        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt
        )

        answer = response.response.strip()

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

#This tools help us to check if the robot has completed the subtask/task
def analyze_subtask_progress(subtask: str, obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking progress for subtask: {subtask}")
        _,_,img_base64, wrist_img_base64, """side_img_base64, robot_img_base64,front_img_base64""" = _obs_to_imgs(obs, args)

        #This prompt is a bit weak, if a task is not detailed e.g "Pick up bowl" this prompt will return no, but we can observe that the robot has completed task. However with a task e.g "Pick up bowl and hold it" the VLM returns done. 
        prompt = (
            f"You are an expert visual inspection system for the robot Franka Emika Panda.\n"
            f"Your job is to determine if the subtask '{subtask}' is COMPLETED.\n"
            
            f"INSTRUCTION TRANSLATION RULES:\n"
            f"- If the task says 'pick', 'lift', or 'grab' -> Check for the **HOLDING STATE**.\n"
            f"- If the task says 'place', 'put', or 'drop' -> Check for the **RELEASED STATE**.\n"

            f"VISUAL CRITERIA FOR SUCCESS:\n"
            f"1. HOLDING STATE (Pick/Grab): The object is firmly clamped inside the gripper. "
            f"IMPORTANT: If the robot is holding the object, the task is DONE, even if the arm is still moving.\n" 
            
            f"2.RELEASED STATE (Place/Put): The gripper is open and explicitly separated from the object. "
            f"The object is resting on the target surface.\n\n"
            
            f"3.OPEN/CLOSE: The mechanism is in its final state.\n"

            f"Analyze the image and answer with strictly one of the following options:\n"
            f"done - if the goal state is achieved.\n"
            f"no - if the goal state is not achieved.\n"
        )
        #Sending necessary image to VLM for checking if a task is done or not.
        all_image_data = [
        img_base64,       
        wrist_img_base64, 
        """ 
        side_img_base64,   
        robot_img_base64,  
        front_img_base64   
        """
        ]

        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            images= all_image_data
        )

        # --- DEBUGGING BLOCK ---

        try:
            logging.info(f"FULL RAW RESPONSE: {response}") 
            
            if hasattr(response, 'finish_reason'):
                logging.info(f"Finish Reason: {response.finish_reason}")
        except:
            pass
        # --------------------------------

        answer = response.response
        logging.info(f"Subtask progress response: {answer}")

        if "done" in answer.lower():
            status = "done"
        elif "no" in answer.lower():
            status = "no"
        else:
            status = "error"

        return {"status": status}

    except Exception as e:
        logging.error(f"Failed to analyze subtask progress: {e}")
        return {"status": "error", "message": str(e)}

#This tool talks with OpenPI VLA, making the robot to move.
def run_robot_action(env, obs, prompt, args, Client_openpi):
    try:
        steps = 0
        max_steps = 250
        vlm_check = 14
        mod = 0
        while steps < max_steps:
            img, wrist_img,*_ = _obs_to_imgs(obs, args)
            cv2.imshow("env", img)
            cv2.imshow("wrist", wrist_img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
                "prompt": str(prompt),
            }

            actions = Client_openpi.infer(element)["actions"]
            chunk_steps = min(args.replan_steps, len(actions))
            for i in range(chunk_steps):
                obs,reward, done, info = env.step(actions[i].tolist())
                steps += 1
                
            mod +=1

            #If modulo with mod and vlm_check returns 0 rest we check with analyze_subtask_progress()
            if mod % vlm_check == 0:
                print("Checking status..")
                progress = analyze_subtask_progress(prompt, obs, args)
                if progress.get("status") == "done":
                    return {"status": "success","obs" :obs}
                elif progress.get("status") == "no":
                    print("We have been using ",steps," steps")
                    continue
                elif progress.get("status") == "error":
                    logging.warning(f"Subtask '{prompt}' failed in VLM check.")
                    return {"status": "error", "obs": obs}
        
        logging.warning(f"Reached Max_steps")
        return {"status": "incomplete", "obs": obs,}

    except Exception as e:
        logging.error(f"Error during run_robot_action: {e}")
        return {"status": "error", "message": str(e), "obs": obs}

#The big tool which executes all of the tools above
def execute_task_with_vlm(task_description: str, env, obs, args: Args, Client_openpi) -> dict:

    #Starts with analyzing if a task is possible or not 
    feasibility = analyze_scene_for_task(task_description, obs, args)
    if feasibility.get("status") != "success":
        logging.error("Scene analysis failed.")
        return {"status": "not_feasible", "obs": obs} 
    
    #Checks if a task is needed to be divided into subtasks
    r = split_task_into_subtasks(task_description)
    subtasks = r.get("subtasks", [task_description])
    logging.info(f"Identified {len(subtasks)} subtasks: {subtasks}")
    logging.info(f"Starting execution for task: '{subtasks}'")
   
   # We run the robot for each subtask
    #subtask_results = []
    for subtask in subtasks:
        logging.info(f"Executing subtask: '{subtask}'")
        result = run_robot_action(env, obs, subtask, args,Client_openpi)
        obs = result.get("obs", obs)
        #subtask_results.append(result)
    return result

def eval_libero(args: Args) -> None:
    # Needed for the simulation, otherwise bullshit
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(0)    
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    obs = env.reset()

    # Here begins the real thing   
    client_OpenPi = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    while True:
        custom_prompt = input("Enter your prompt (type exit to stop): ")
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
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, }
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
