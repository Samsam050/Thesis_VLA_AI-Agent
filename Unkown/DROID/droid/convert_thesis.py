import shutil
import os
import glob
from pathlib import Path
import h5py
import numpy as np
import tyro
import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# --- CONFIGURATION ---
REPO_NAME = "panda_droid_converted"
ROBOT_TYPE = "panda"
FPS = 15

# === CAMERA MAPPING ===
# Based on your provided IDs:
# 1. varied_camera_1 (242422301956) -> Main "image" (Exterior/Third-Person)
# 2. hand_camera     (19443010513EA12E00) -> "wrist_image"
# We append "_left" because DROID saves stereo cameras as left/right, and you usually want left.
EXTERIOR_CAMERA_KEY = "observations/videos/242422301956_left" 
WRIST_CAMERA_KEY    = "observations/videos/19443010513EA12E00_left"

def extract_frames_from_video_bytes(video_bytes):
    """Writes video bytes to a temp file and reads frames back."""
    temp_path = f"temp_decoding_{os.getpid()}.mp4" # Use PID to avoid collision
    
    # 1. Write bytes to temp file
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
        
    # 2. Read frames using OpenCV
    cap = cv2.VideoCapture(temp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return np.array(frames)

def main(
    root_data_dir: str = "collected_data", 
    *, 
    push_to_hub: bool = False
):
    root_path = Path(root_data_dir)
    # Find all HDF5 files recursively
    files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))
    
    if not files:
        print(f"No HDF5 files found in {root_path}")
        return

    print(f"Found {len(files)} episodes. Starting conversion...")

    # 1. Setup LeRobot Dataset
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3), 
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,), # 7 joints + 1 gripper
                "names": ["joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5", "joint_pos_6", "joint_pos_7", "gripper_width"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5", "joint_vel_6", "joint_vel_7", "gripper_action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    count = 0
    for file_path in files:
        print(f"Processing: {file_path.name}")
        try:
            with h5py.File(file_path, "r") as f:
                
                # --- 1. EXTRACT STATE & ACTION ---
                
                # State: (N, 7)
                joint_pos = f['observations/robot_state/joint_positions'][:]
                # Gripper: (N,) -> reshape to (N, 1)
                gripper_pos = f['observations/robot_state/gripper_position'][:]
                gripper_pos = gripper_pos.reshape(-1, 1)
                
                state_data = np.concatenate([joint_pos, gripper_pos], axis=1)

                # Action: (N, 7)
                joint_vel = f['action/joint_velocity'][:]
                # Action Gripper: (N,) -> reshape to (N, 1)
                gripper_act = f['action/gripper_position'][:]
                gripper_act = gripper_act.reshape(-1, 1)
                
                action_data = np.concatenate([joint_vel, gripper_act], axis=1)

                # --- 2. EXTRACT VIDEO ---
                
                # Exterior Camera (Varied Camera)
                if EXTERIOR_CAMERA_KEY in f:
                    video_bytes = f[EXTERIOR_CAMERA_KEY][()]
                    ext_frames = extract_frames_from_video_bytes(video_bytes)
                else:
                    raise KeyError(f"Exterior camera key {EXTERIOR_CAMERA_KEY} not found.")

                # Wrist Camera (Hand Camera)
                if WRIST_CAMERA_KEY in f:
                    video_bytes = f[WRIST_CAMERA_KEY][()]
                    wrist_frames = extract_frames_from_video_bytes(video_bytes)
                else:
                    print(f"Warning: Wrist camera key {WRIST_CAMERA_KEY} not found. Skipping wrist image.")
                    wrist_frames = None

                # --- 3. SYNCHRONIZE & SAVE ---
                # Ensure all streams have the same length (take the minimum)
                num_steps = min(len(state_data), len(ext_frames))
                if wrist_frames is not None:
                    num_steps = min(num_steps, len(wrist_frames))
                
                # Optionally filter out the very first few frames if robot wasn't moving?
                # For now, we keep everything.

                print(f"  > Merging {num_steps} frames")

                for i in range(num_steps):
                    
                    # Resize images to 256x256
                    img = cv2.resize(ext_frames[i], (256, 256))
                    
                    frame_dict = {
                        "image": img,
                        "state": state_data[i],
                        "actions": action_data[i],
                    }
                    
                    if wrist_frames is not None:
                         wrist_img = cv2.resize(wrist_frames[i], (256, 256))
                         frame_dict["wrist_image"] = wrist_img
                    
                    dataset.add_frame(frame_dict)

                dataset.save_episode()
                count += 1

        except Exception as e:
            print(f"Failed to convert {file_path.name}: {e}")

    print(f"\nSUCCESS! Saved {count} episodes to {output_path}")
    print("Visualize with:")
    print(f"lerobot-visualize-dataset --repo-id {REPO_NAME} --local-files-only 1")
    
    if push_to_hub:
        print("Pushing to Hub...")
        dataset.push_to_hub(tags=["droid", "panda", "video_decoding"], private=True)

if __name__ == "__main__":
    tyro.cli(main)