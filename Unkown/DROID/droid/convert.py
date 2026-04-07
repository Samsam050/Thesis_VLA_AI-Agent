"""
old 
"""

import shutil
import os
import glob
import h5py
import json
import tempfile
import cv2
import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# --- SETTINGS ---
DATA_DIR = "collected_data"
LABEL_FILE = "dataset_labels.json"
REPO_NAME = "thesis/pi05_dataset_v3"  # <--- V3 to be safe
ROBOT_TYPE = "panda"
FPS = 15
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

def extract_frames_from_video_blob(video_bytes):
    frames = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_name = tmp.name

    cap = cv2.VideoCapture(tmp_name)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        if frame.shape[0] != VIDEO_HEIGHT or frame.shape[1] != VIDEO_WIDTH:
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
        frames.append(frame)
    
    cap.release()
    os.remove(tmp_name)
    return frames

def main(push_to_hub: bool = False):
    # 1. Clean up
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"🗑️  Cleaning up {output_path}")
        shutil.rmtree(output_path)

    # 2. Load Labels
    if not os.path.exists(LABEL_FILE):
        print(f"❌ Error: {LABEL_FILE} missing.")
        return
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)

    # 3. Find Files
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.h5"), recursive=True)
    all_files.sort()
    
    if not all_files:
        print("❌ No data found.")
        return

    print(f"🤖 Initializing Dataset for {ROBOT_TYPE}...")
    
    # 4. Create Dataset
    # We use "task" here because LeRobot demands it.
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (VIDEO_HEIGHT, VIDEO_WIDTH, 3), 
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["action"],
            },
            "task": {
                "dtype": "string",
                "shape": (1,),  
                "names": ["task"]
            }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    count = 0
    skipped = 0
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        task_prompt = labels.get(filename, "unknown task")
        
        if task_prompt == "unknown task":
            skipped += 1
            continue

        try:
            with h5py.File(filepath, "r") as f:
                # --- ACTIONS (Force Float32) ---
                arm_actions = f["action/joint_position"][:]
                if "action/gripper_position" in f:
                    grip_actions = f["action/gripper_position"][:]
                else:
                    grip_actions = np.zeros((len(arm_actions), 1))
                if grip_actions.ndim == 1: grip_actions = grip_actions.reshape(-1, 1)
                
                # MERGE & CAST
                full_actions = np.concatenate([arm_actions, grip_actions], axis=1).astype(np.float32)

                # --- STATES (Force Float32) ---
                if "observations/robot_state/joint_positions" in f:
                    arm_states = f["observations/robot_state/joint_positions"][:]
                else:
                    arm_states = np.zeros_like(arm_actions)
                if "observations/robot_state/gripper_position" in f:
                    grip_states = f["observations/robot_state/gripper_position"][:]
                else:
                    grip_states = np.zeros((len(arm_states), 1))
                if grip_states.ndim == 1: grip_states = grip_states.reshape(-1, 1)
                
                # MERGE & CAST
                full_states = np.concatenate([arm_states, grip_states], axis=1).astype(np.float32)

                # --- VIDEO ---
                vid_keys = list(f["observations/videos"].keys())
                primary_cam = vid_keys[0]
                video_blob = f[f"observations/videos/{primary_cam}"][()]
                frames = extract_frames_from_video_blob(video_blob)
                
                min_len = min(len(frames), len(full_actions), len(full_states))
                
                for i in range(min_len):
                    dataset.add_frame({
                        "image": frames[i],
                        "state": full_states[i],
                        "action": full_actions[i],
                       "task": step["language_instruction"].decode(),
                    })
                
                dataset.save_episode()
                count += 1
                if count % 10 == 0:
                    print(f"   [{count}/{len(all_files)}] Processed...")

        except Exception as e:
            print(f"❌ Failed {filename}: {e}")

    print("\n" + "="*40)
    print(f"🎉 SUCCESS! Converted {count} episodes.")
    print(f"📁 LOCATION: {HF_LEROBOT_HOME / REPO_NAME}")
    print("="*40)

if __name__ == "__main__":
    tyro.cli(main)