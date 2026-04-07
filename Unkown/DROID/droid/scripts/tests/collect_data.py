import os
import datetime
import glob
import sys
from droid.controllers.haptic_touch_controller import HapticTouchPolicy
from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import collect_trajectory

# CONFIGURATION 
TASK_DESCRIPTION = "Put fruit in bowl"
TASK_FOLDER = "put_fruit_in_bowl"
DATA_DIR = "collected_data"
SAVE_IMAGES = True 

output_dir = os.path.join(DATA_DIR, TASK_FOLDER)
os.makedirs(output_dir, exist_ok=True)

def get_next_episode_idx(output_dir):
    """Scans the directory to find the next available episode number."""
    # Look for files like "episode_0_*.h5", "episode_1_*.h5"
    existing_files = glob.glob(os.path.join(output_dir, "episode_*_*.h5"))
    
    if not existing_files:
        return 0
    
    indices = []
    for f in existing_files:
        try:
            filename = os.path.basename(f)
            idx = int(filename.split('_')[1])
            indices.append(idx)
        except (IndexError, ValueError):
            continue
            
    return max(indices) + 1 if indices else 0

def main():
    print(f"Initializing DROID Environment for task: {TASK_DESCRIPTION}")
    
    episode_idx = get_next_episode_idx(output_dir)
    print(f"Auto-detected next episode index: {episode_idx}")
    
    env = RobotEnv()
    controller = HapticTouchPolicy()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{episode_idx}_{timestamp}.h5"
    full_path = os.path.join(output_dir, filename)

    print(f"   READY TO RECORD EPISODE {episode_idx}")
    print(f"   Recording will start as soon as you move the robot...")
    print(f"   (Data will be saved to: {filename})")

    # This function blocks until you press 's' or 'q'
    info = collect_trajectory(
        env=env,
        controller=controller,
        save_filepath=full_path,    
        save_images=SAVE_IMAGES,   
        wait_for_controller=True, 
        metadata={
            "task": TASK_DESCRIPTION, 
            "robot": "Franka Panda",
            "date": timestamp
        }
    )

    # 4. Check status
    if info['success']:
        print(f"SUCCESS! Episode {episode_idx} saved.")
        print(f"SAVED TO PATH: {os.path.abspath(full_path)}")
    elif info['failure']:
        print(f"FAILURE. Episode {episode_idx} marked as failed.")
    else:
        print("Episode finished without explicit tag.")
        
if __name__ == "__main__":
    main()