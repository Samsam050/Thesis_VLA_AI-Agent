import os
import glob
import json

DATA_DIR = "collected_data"
JSON_FILE = "dataset_labels.json"

def main():
    print("🕵️ STARTING FINAL DATA AUDIT...")
    
    # 1. Load the Labels
    if not os.path.exists(JSON_FILE):
        print("❌ CRITICAL ERROR: 'dataset_labels.json' is missing!")
        return
        
    with open(JSON_FILE, 'r') as f:
        labels = json.load(f)
    
    print(f"📄 Loaded {len(labels)} labels from JSON.")

    # 2. Find ALL episodes on disk
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.h5"), recursive=True)
    total_files = len(all_files)
    print(f"📂 Found {total_files} episode files on disk.")
    
    # 3. Compare
    missing_labels = []
    empty_labels = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Check if filename is in the JSON keys
        if filename not in labels:
            missing_labels.append(filename)
        else:
            # Check if the label is empty string or None
            prompt = labels[filename]
            if not prompt or prompt.strip() == "":
                empty_labels.append(filename)

    # 4. Report Results
    print("\n" + "="*40)
    print("AUDIT RESULTS")
    print("="*40)
    
    if len(missing_labels) == 0 and len(empty_labels) == 0:
        print("✅ SUCCESS! Every single episode has a label.")
        print(f"   Total Data: {total_files} episodes ready for conversion.")
        print("   Action: You can run 'uv run convert_with_labels.py' now.")
    else:
        print(f"❌ ISSUES FOUND: {len(missing_labels) + len(empty_labels)} problems.")
        
        if len(missing_labels) > 0:
            print(f"\n⚠️ The following {len(missing_labels)} files have NO label:")
            for name in missing_labels:
                print(f"   - {name}")
                
        if len(empty_labels) > 0:
            print(f"\n⚠️ The following {len(empty_labels)} files have EMPTY labels:")
            for name in empty_labels:
                print(f"   - {name}")
                
        print("\nAction: Run 'uv run speed_tagger_v4.py' again to catch these missing files.")

if __name__ == "__main__":
    main()