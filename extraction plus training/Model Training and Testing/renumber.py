import os
import shutil
import random

# Define folder paths
AGE_DATASET_FOLDER = r"C:\EEE\Age Dataset\SortedFiles\Age_dataset"
NEW_DATASET_FOLDER = r"C:\EEE\BESD_Split\train"  # New dataset folder

# --- Step 1: Rename existing Age Dataset files consistently ---

def rename_existing_files(folder):
    male_files = []
    female_files = []
    
    # List all .wav files and separate by gender based on 'male'/'female' in filename (case-insensitive)
    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            if "male" in filename.lower():
                male_files.append(filename)
            elif "female" in filename.lower():
                female_files.append(filename)
    
    # Sort lists for consistent ordering
    male_files.sort()
    female_files.sort()
    
    # Rename male files sequentially
    for i, old_name in enumerate(male_files, 1):
        new_name = f"male_child_{i}.wav"
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} -> {new_name}")
        
    # Rename female files sequentially
    for i, old_name in enumerate(female_files, 1):
        new_name = f"female_child_{i}.wav"
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} -> {new_name}")
    
    return len(male_files), len(female_files)

print("Renaming existing Age Dataset files...")
num_existing_males, num_existing_females = rename_existing_files(AGE_DATASET_FOLDER)
print(f"Existing Age Dataset: {num_existing_males} male files, {num_existing_females} female files.\n")

# --- Step 2: Process new dataset files and prepare new names ---
def prepare_new_files(new_folder, start_male, start_female):
    male_files = []
    female_files = []
    
    # List new dataset .wav files and separate by gender (using substring matching)
    for filename in os.listdir(new_folder):
        if filename.lower().endswith(".wav"):
            if "male" in filename.lower():
                male_files.append(filename)
            elif "female" in filename.lower():
                female_files.append(filename)
    
    male_files.sort()
    female_files.sort()
    
    new_files_info = []  # List of tuples (source_path, new_filename)
    
    # Rename male files starting after current count
    for i, old_name in enumerate(male_files, start=start_male + 1):
        new_name = f"male_child_{i}.wav"
        src_path = os.path.join(new_folder, old_name)
        new_files_info.append((src_path, new_name))
    
    # Rename female files starting after current count
    for i, old_name in enumerate(female_files, start=start_female + 1):
        new_name = f"female_child_{i}.wav"
        src_path = os.path.join(new_folder, old_name)
        new_files_info.append((src_path, new_name))
    
    return new_files_info

print("Preparing new dataset files for merging...")
new_files = prepare_new_files(NEW_DATASET_FOLDER, num_existing_males, num_existing_females)
print(f"Prepared {len(new_files)} new files.\n")

# --- Step 3: Copy the new files into the Age Dataset folder ---
for src_path, new_filename in new_files:
    dest_path = os.path.join(AGE_DATASET_FOLDER, new_filename)
    shutil.copy(src_path, dest_path)
    print(f"Copied {os.path.basename(src_path)} -> {new_filename}")

print("\n✅ All new dataset files have been copied to the Age Dataset folder with consistent naming.")
