import h5py
import numpy as np

# REPLACE with your actual file path
path = "collected_data/put_fruit_in_bowl/episode_0_20260125_203630.h5" 


with h5py.File(path, 'r') as f:
    print("--- VIDEO KEYS ---")
    if "observations/videos" in f:
        f["observations/videos"].visit(lambda x: print(x))
    
    print("\n--- ACTION KEYS ---")
    if "action" in f:
        f["action"].visit(lambda x: print(x))

    print("\n--- STATE KEYS ---")
    if "observations/robot_state" in f:
        f["observations/robot_state"].visit(lambda x: print(x))