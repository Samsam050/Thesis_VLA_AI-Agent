import h5py
import numpy as np
from pathlib import Path

# --- CONFIG ---
DATA_DIR = "collected_data/put_fruit_in_bowl" 
# --------------

def print_structure(name, obj):
    """Recursive function to print everything in the file"""
    if isinstance(obj, h5py.Dataset):
        # If it's a dataset, print its shape and type
        print(f"  📂 {name}  ->  {obj.shape} ({obj.dtype})")
        
        # If it looks like text, try to print the content
        if "instruction" in name or obj.dtype.kind == 'S' or obj.dtype.kind == 'O':
            try:
                # Try decoding as a single string
                content = obj[()]
                if isinstance(content, bytes):
                    print(f"     📝 FOUND TEXT: '{content.decode('utf-8')}'")
                else:
                    print(f"     📝 FOUND CONTENT: {content}")
            except:
                pass

    elif isinstance(obj, h5py.Group):
        # If it's a folder, just print the name
        pass # We rely on visititems to print names

def inspect_file(file_path):
    print(f"\n========================================================")
    print(f" 🕵️ LOOKING FOR TEXT IN: {file_path.name}")
    print(f"========================================================")
    
    try:
        with h5py.File(file_path, "r") as f:
            
            # 1. CHECK ROOT ATTRIBUTES (Metadata often hides here)
            print("--- ROOT ATTRIBUTES ---")
            if len(f.attrs) > 0:
                for key, value in f.attrs.items():
                    print(f"  🔹 {key}: {value}")
            else:
                print("  (No root attributes found)")
            
            # 2. CHECK ALL KEYS (Recursive search)
            print("\n--- ALL DATA KEYS ---")
            f.visititems(print_structure)
            
            # 3. SPECIFIC CHECK FOR COMMON INSTRUCTION KEYS
            print("\n--- INSTRUCTION CHECK ---")
            candidates = ["language_instruction", "instruction", "task_description", "metadata/language_instruction"]
            found_any = False
            for key in candidates:
                if key in f:
                    print(f"  ✅ FOUND '{key}'!")
                    found_any = True
                else:
                    print(f"  ❌ '{key}' not found.")
            
            if not found_any:
                print("\n⚠️ CONCLUSION: No standard text keys found.")
                print("   We will probably have to use the FOLDER NAME.")
            else:
                print("\n✅ CONCLUSION: We found the text! The script will work.")

    except Exception as e:
        print(f"Error reading file: {e}")

# Run on the last file found
root = Path(DATA_DIR)
files = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))

if files:
    inspect_file(files[-1]) # Check the last file
else:
    print("No files found.")