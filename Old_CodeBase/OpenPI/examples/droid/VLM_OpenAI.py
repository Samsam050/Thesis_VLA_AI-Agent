import contextlib
import logging
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
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from openai import OpenAI
import base64
import ast
import io

client = OpenAI()

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "<your_camera_id>"  # e.g., "24259877"
    right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    wrist_camera_id: str = "<your_camera_id>"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


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


def encode_rgb_image(image_np: np.ndarray) -> str:
    """Helper to convert RGB numpy array to Base64 string."""

    # Ensure uint8 (standard image format)
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # Convert to PIL and save to memory buffer
    pil_img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def analyze_scene_for_task(task_description: str, curr_obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking if task '{task_description}' is possible...")
        left_camera_id = obs["observation/exterior_image_1_left"]
        wrist_camera_id = obs["observation/wrist_image_left"]
        _, buf_ext = cv2.imencode(".jpg", left_camera_id)
        _, buf_wrist = cv2.imencode(".jpg", wrist_camera_id)
        ext_b64 = base64.b64encode(buf_ext).decode("utf-8")
        wrist_b64 = base64.b64encode(buf_wrist).decode("utf-8")
     
    

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
                            "image_url": f"data:image/jpeg;base64,{ext_b64}"}
                            ,
                            {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{wrist_b64}"
                            }]}]
        )

        answer = response.output_text
        logging.info(f"VLM response: {answer}")

        return {"status": "success", "analysis": answer}

    except Exception as e:
        logging.error(f"Failed to analyze scene: {e}")
        return {"status": "error", "message": str(e)}

def analyze_subtask_progress(subtask: str, curr_obs: dict, args:Args) -> dict:
    try:
        logging.info(f"Checking progress for subtask: {subtask}")
    
        left_camera_id = obs["observation/exterior_image_1_left"]
        wrist_camera_id = obs["observation/wrist_image_left"]
        _, buf_ext = cv2.imencode(".jpg", left_camera_id)
        _, buf_wrist = cv2.imencode(".jpg", wrist_camera_id)
        ext_b64 = base64.b64encode(buf_ext).decode("utf-8")
        wrist_b64 = base64.b64encode(buf_wrist).decode("utf-8")


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

def run_robot_action(env, curr_obs, prompt, args, Client_openpi):
    try:
        t = 0
        max_steps = 250
        vlm_c = 14
        C = 0
        while t < max_steps:

            element = {
                "observation/exterior_image_1_left": ex_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": str(prompt),
            }

            actions = Client_openpi.infer(element)["actions"]
            chunk_steps = min(args.replan_steps, len(actions))
            for i in range(chunk_steps):
                obs,reward, done, info = env.step(actions[i].tolist())
                t += 1
                
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
        
        logging.warning(f"Reached Max_steps")
        return {"status": "incomplete", "obs": obs,}

    except Exception as e:
        logging.error(f"Error during run_robot_action: {e}")
        return {"status": "error", "message": str(e), "obs": obs}


def execute_task_with_vlm(task_description: str, env, curr_obs, args: Args, Client_openpi) -> dict:

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

    return result



def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, wrist_image = None, None
    for key in image_observations:
        #Checking for the left camera
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        #Checking for the wrist camera    
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
