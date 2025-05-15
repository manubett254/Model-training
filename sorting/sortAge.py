import os
import shutil
import pandas as pd

# Define paths
audio_folder = 'C:/EEE/en/en/validFiles'  # Folder with sorted audio clips
csv_file_path = 'C:/EEE/en/en/validated_distinct.csv'  # Metadata file
output_folder = 'C:/EEE/en/en/sortedFiles'  # New folder for sorted files

# Read CSV and filter valid rows
df = pd.read_csv(csv_file_path)
df_filtered = df.dropna(subset=['path', 'age', 'gender'])

# Track files moved
files_moved = 0

# Iterate over files and move them to respective folders
for index, row in df_filtered.iterrows():
    file_name = str(row['path']).strip().lower() + ".mp3"  # Append .mp3 extension
    age_group = str(row['age']).replace(" ", "_")  # Remove spaces in age
    gender = str(row['gender']).lower()

    # Define destination folder
    destination_folder = os.path.join(output_folder, gender, age_group)
    os.makedirs(destination_folder, exist_ok=True)  # Ensure folder exists

    old_path = os.path.join(audio_folder, file_name)
    new_path = os.path.join(destination_folder, file_name)

    # Check if the file exists before moving
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        files_moved += 1
    else:
        print(f"⚠️ File not found: {old_path}")  # Debug missing files

print(f"✅ {files_moved} files have been sorted into folders!")
