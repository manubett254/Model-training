import os
import shutil
import random

# Paths (Change these to match your actual paths)
final_dataset_folder = r"C:\EEE\Big dataset\clip\finalDataset"
backup_folder = r"C:\EEE\Big dataset\backup_male"

# Ensure backup folder exists
os.makedirs(backup_folder, exist_ok=True)

# Get all male files
male_files = [f for f in os.listdir(final_dataset_folder) if f.startswith("male")]

# Randomly select 1,040 files
selected_males = random.sample(male_files, 1040)

# Files to be moved to backup
files_to_move = set(male_files) - set(selected_males)

# Move extra files to backup
for file in files_to_move:
    shutil.move(os.path.join(final_dataset_folder, file), os.path.join(backup_folder, file))

print(f"Selected {len(selected_males)} male files.")
print(f"Moved {len(files_to_move)} male files to backup.")
