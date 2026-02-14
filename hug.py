from huggingface_hub import HfApi


from datasets import load_dataset

# 1. Load your dataset from Hugging Face
print("Loading dataset info...")
ds = load_dataset("shuooru/franka_robot_finetune", split="train")

# 2. Check if it is stored as Episodes or Individual Steps
print(f"Number of rows in dataset: {len(ds)}")

# IF the rows = 100, then each row is an 'Episode' containing many steps.
# We need to sum the length of each episode.

total_steps = 0

# We look at the first row to see column names. 
# Usually 'images', 'observation', or 'action' holds the steps.
sample_row = ds[0]

# Try to find a column that is a list (sequence of frames)
list_columns = [key for key, val in sample_row.items() if isinstance(val, (list, dict))]

if len(ds) == 100:
    print("Dataset seems to be stored as Episodes. Counting frames inside...")
    # Sum the length of the first available list column (e.g. 'steps' or 'images')
    # Note: Adjust 'images' if your column is named differently (e.g. 'steps')
    for item in ds:
        # We assume the episode length is the length of the primary data column
        # Try 'episode_length' if it exists, otherwise count the list
        if 'episode_length' in item:
             total_steps += item['episode_length']
        else:
             # Fallback: Count length of the first list found (e.g. actions/images)
             first_list_col = list_columns[0] 
             total_steps += len(item[first_list_col])
else:
    # If len(ds) is huge (e.g. 30,000), then it is already flattened into steps
    total_steps = len(ds)

print("-" * 30)
print(f"Total STEPS (Frames) found: {total_steps}")