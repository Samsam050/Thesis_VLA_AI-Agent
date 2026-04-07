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
    right_camera_id: str = ""  # e.g., "24514023"
    wrist_camera_id: str = "19443010513EA12E00"  # e.g., "13062452"

    # Policy parameters
    external_camera: Optional[str] = "left" # which external camera should be fed to the policy, choose from ["left", "right"]
    

    # Rollout parameters
    max_timesteps: int = 10000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "130.243.124.173"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
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

class Display:
    def __init__(self):
        # Must be initialized in the main thread
        cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)
        cv2.startWindowThread() 
    def show(self, frame):
        if frame is not None:
            # Policy uses RGB, OpenCV needs BGR
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Robot View", bgr_frame)
            cv2.waitKey(1)
            cv2.pollKey()


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")
    time.sleep(3.0)
    display = Display()

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

                img_left = curr_obs["left_image"]
                img_wrist = curr_obs["wrist_image"]
                h_target = 480
                w_left = int(img_left.shape[1] * (h_target / img_left.shape[0]))
                w_wrist = int(img_wrist.shape[1] * (h_target / img_wrist.shape[0]))

                disp_left = cv2.resize(img_left, (w_left, h_target))
                disp_wrist = cv2.resize(img_wrist, (w_wrist, h_target))
            
                display_frame = np.concatenate([disp_left, disp_wrist], axis=1)
                display.show(display_frame)
                
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
                    print(f"DEBUG: The server returned shape: {pred_action_chunk.shape}")
                    assert pred_action_chunk.shape == (15, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
            
                #print(action[-1].item())
                # Binarize gripper action
               #action=action.copy()
            
                #action[:-1] = action[:-1]*Hz
                actions_from_chunk_completed += 1
                #raw_gripper = action[-1]
                #real_gripper = (raw_gripper * 0.5) + 0.4
                #print(f"Raw: {raw_gripper:.4f} -> Real: {real_gripper:.4f}")    
                print(action[-1])
                if action[-1].item() > 0.5:
                #if real_gripper > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                    print("close")
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                    print("open")

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)
               

                env.step(action)
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: Union[str, float, None] = None
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
