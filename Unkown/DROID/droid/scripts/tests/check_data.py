import os
import glob
import h5py
import numpy as np

DATA_DIR = "collected_data"

def check_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Check for Action Data
            if 'action/joint_position' not in f:
                return " MISSING ACTIONS"
            
            # 2. Check for Video Data
            if 'observations/videos' not in f:
                return " MISSING VIDEO GROUP"
            
            video_keys = list(f['observations/videos'].keys())
            if len(video_keys) == 0:
                return " VIDEO GROUP EMPTY"

            # 3. Check Episode Length (Actions)
            actions = f['action/joint_position'][:]
            episode_len = len(actions)
            if episode_len < 10:
                return f"TOO SHORT ({episode_len} steps)"

            # 4. Check Video Integrity (Scalar vs Array)
            # We pick the first camera
            cam_name = video_keys[0]
            video_dataset = f[f'observations/videos/{cam_name}']
            
            # If shape is empty tuple (), it is a Scalar (Compressed Video Blob)
            if video_dataset.shape == ():
                # We can't check 'len', but we can check if it has data
                # A valid scalar dataset usually has a .dtype of object or uint8
                if video_dataset.dtype == 'O' or video_dataset.size > 0:
                    return " OK (Compressed Video)"
                else:
                    return "EMPTY VIDEO BLOB"
            else:
                # It is a raw array of frames
                video_len = video_dataset.shape[0]
                if abs(episode_len - video_len) > 5:
                     return f" SYNC ISSUE (Action: {episode_len}, Video: {video_len})"
                return " OK (Raw Frames)"

    except OSError:
        return "❌ CORRUPT FILE (Cannot Open)"
    except Exception as e:
        return f"❌ ERROR: {str(e)}"

def main():
    print(f" Starting Health Check V3 (Scalar/Blob Support)...\n")
    
    tasks = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    total_files = 0
    bad_files = 0
    
    for task in tasks:
        task_path = os.path.join(DATA_DIR, task)
        files = glob.glob(os.path.join(task_path, "*.h5"))
        print(f" Scanning '{task}' ({len(files)} episodes)...")
        
        for file in files:
            status = check_file(file)
            total_files += 1
            if "OK" not in status:
                bad_files += 1
                print(f"   -> {os.path.basename(file)}: {status}")
                
    print(f"\n" + "="*30)
    print(f" SCAN COMPLETE")
    print(f"   Total Episodes: {total_files}")
    if bad_files == 0:
        print(f"   Status:  PERFECT HEALTH")
    else:
        print(f"   Status: {bad_files} ISSUES FOUND")

if __name__ == "__main__":
    main()