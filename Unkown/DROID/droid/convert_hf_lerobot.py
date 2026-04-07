"correct one after som struggles"


import h5py
import numpy as np
import cv2
import os
import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from tqdm import tqdm
import tyro

# --- CONFIGURATION ---
REPO_NAME = "panda_droid_finetune_oru"  # Changed name to avoid conflicts with previous attempts
TASK_DESCRIPTION = "put_fruit_in_bowl"
FPS = 15

# === CAMERA CONFIG ===
# We map your specific serial numbers to the standard LeRobot/OpenPI names.
# Key: Your specific ID (from your output)
# Value: The standardized name Pi0 expects
EXTERIOR_ID = "242422301956_left"
WRIST_ID    = "19443010513EA12E00_left"

def decode_video_bytes(video_bytes):
    """Decodes raw MP4 bytes into a list of numpy images (RGB)."""
    if video_bytes is None or len(video_bytes) == 0:
        return None
        
    # Write bytes to a temporary file
    temp_path = f"temp_{os.getpid()}.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV is BGR, LeRobot needs RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return frames

def resize_image(img, target_shape=(320, 180)):
    return cv2.resize(img, target_shape)

def main(data_dir: str = "collected_data"):
    # 1. Setup LeRobot Dataset
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_pos"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_pos"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    root_path = Path(data_dir)
    files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))
    print(f"Found {len(files)} episodes.")

    for file_path in tqdm(files, desc="Converting"):
        try:
            with h5py.File(file_path, "r") as f:
                
                # --- CHECK 1: Do we have actions? ---
                if "action/joint_velocity" not in f:
                    print(f"Skipping {file_path.name}: No joint_velocity found.")
                    break
                    
                joint_vel = f['action/joint_velocity'][:]
                if len(joint_vel) == 0:
                    print(f"Skipping {file_path.name}: joint_velocity is empty.")
                    break

                # --- CHECK 2: Do we have images? ---
                # Construct full paths
                ext_key = f"observations/videos/{EXTERIOR_ID}"
                wrist_key = f"observations/videos/{WRIST_ID}"
                
                image_streams = {}
                
                if ext_key in f:
                    data = f[ext_key][()]
                    frames = decode_video_bytes(data)
                    if frames and len(frames) > 0:
                        image_streams["exterior_image_1_left"] = frames
                
                if wrist_key in f:
                    data = f[wrist_key][()]
                    frames = decode_video_bytes(data)
                    if frames and len(frames) > 0:
                        image_streams["wrist_image_left"] = frames

                if not image_streams:
                    print(f"Skipping {file_path.name}: No valid video data found.")
                    break

                # --- LOAD KINEMATICS ---
                joint_pos = f['observations/robot_state/joint_positions'][:]
                gripper_pos = f['observations/robot_state/gripper_position'][:]
                
                # Ensure gripper is (N, 1)
                if len(gripper_pos.shape) == 1:
                    gripper_pos = gripper_pos.reshape(-1, 1)

                # Prepare Actions (Velocity + Gripper Position)
                gripper_act = f['action/gripper_position'][:]
                if len(gripper_act.shape) == 1:
                    gripper_act = gripper_act.reshape(-1, 1)
                
                # Stack them: 7 velocity + 1 gripper = 8
                actions = np.concatenate([joint_vel, gripper_act], axis=1)

                # --- SYNC LENGTHS ---
                # Find the lowest common frame count
                n_frames = len(joint_pos)
                n_frames = min(n_frames, len(actions))
                for key, frames in image_streams.items():
                    n_frames = min(n_frames, len(frames))

                if n_frames == 0:
                    print(f"Skipping {file_path.name}: Resulting episode length is 0.")
                    continue

                # --- WRITE FRAMES ---
                for i in range(n_frames):
                    frame_dict = {
                        "joint_position": joint_pos[i],
                        "gripper_position": gripper_pos[i],
                        "actions": actions[i],
                        "task": TASK_DESCRIPTION
                    }

                    if "exterior_image_1_left" in image_streams:
                        frame_dict["exterior_image_1_left"] = resize_image(image_streams["exterior_image_1_left"][i])
                    
                    if "wrist_image_left" in image_streams:
                        frame_dict["wrist_image_left"] = resize_image(image_streams["wrist_image_left"][i])

                    dataset.add_frame(frame_dict)

                dataset.save_episode()

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("Done! Dataset saved to:", dataset.root)
    print("Visualize with:")
    print(f"lerobot-visualize-dataset --repo-id {REPO_NAME} --local-files-only 1")

if __name__ == "__main__":
    tyro.cli(main)