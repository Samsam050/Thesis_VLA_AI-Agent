import h5py
import cv2
import numpy as np
import argparse
import time

def play_episode(file_path):
    print(f"Inspecting: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 1. Check Metadata
        print("Keys in root:", list(f.keys()))
        
        # Handle the key name fix (observation vs observations)
        root_key = "observation" if "observation" in f else "observations"
        root = f[root_key]
        
        # 2. Check Robot State (The Fix is here!)
        robot_state = root['robot_state']
        print(f"Keys in robot_state: {list(robot_state.keys())}")

        # Check Arm Joints
        if 'joint_positions' in robot_state:
            qpos = robot_state['joint_positions'][:]
            print(f"Arm Joints Shape: {qpos.shape}")
            print("Sample joint positions:", qpos[:3])

        if 'cartesian_position' in robot_state:
            pose = robot_state['cartesian_position'][:]
            print(f"Cartesian Position Shape: {pose.shape}")
            print("Sample cartesian pose:", pose[:3])
        # Check Gripper
        if 'gripper_position' in robot_state:
            gpos = robot_state['gripper_position'][:]
            print(f"Gripper Shape:    {gpos.shape}")
        else:
            print("❌ WARNING: Gripper data missing!")

        # 3. Decode and Play Video
        if 'videos' in root:
            video_group = root['videos']
            print(f"Found cameras: {list(video_group.keys())}")
            
            for cam_name in video_group.keys():
                # We only want to verify, so let's just look at one camera to save time
                if "right" in cam_name: 
                    continue # Skip right eye to be faster
                
                print(f"Playing camera: {cam_name} (Press 'q' to quit)")
                
                video_bytes = video_group[cam_name][()]
                temp_filename = "temp_playback.mp4"
                with open(temp_filename, "wb") as temp_f:
                    temp_f.write(video_bytes)
                
                cap = cv2.VideoCapture(temp_filename)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    #cv2.putText(frame, f"Cam: {cam_name}", (10, 30), 
                                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Convert RGB to BGR for display
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Data Verification', frame_bgr)
                    
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                cap.release()
        else:
            print("ERROR: No 'videos' group found in HDF5 file!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update filename if needed
    FILENAME = "collected_data/put_toy_in_box/episode_23_20260124_233648.h5" 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=FILENAME, help="Path to .h5 file")
    args = parser.parse_args()
    
    play_episode(args.file)