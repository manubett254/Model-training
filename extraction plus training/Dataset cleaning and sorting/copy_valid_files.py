import os
import shutil
import pandas as pd

# Load the CSV file
csv_path = r"C:\EEE\Big dataset\validated_cleaned_distinct.csv"
df = pd.read_csv(csv_path)

# Define source and destination directories
src_folder = r"C:\EEE\Big dataset\clips"
dest_folder = r"C:\EEE\Big dataset\clip\validFiles"

# Ensure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

# Get the list of filenames (add .mp3)
file_list = df['path'].str.strip() + ".mp3"

# Track stats
copied_files = 0
missing_files = 0

# Copy matching files
for file_name in file_list:
    src_path = os.path.join(src_folder, file_name)
    dest_path = os.path.join(dest_folder, file_name)  

    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)  # Copy with metadata
        copied_files += 1
    else:
        print(f"🚨 File not found: {src_path}")
        missing_files += 1

print(f"✅ Done! {copied_files} files copied. {missing_files} files were missing.")
