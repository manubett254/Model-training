import os
import shutil

# Define paths
source_folder = r"C:\EEE\Big dataset\clip\renamedFiles"
destination_folder = r"C:\EEE\Big dataset\clip\finalDataset"

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Define male age groups to include
allowed_male_ages = ["twenties", "thirties", "forties", "fifties", "sixties", "seventies", "eighties"]

# Process files
for filename in os.listdir(source_folder):
    if filename.endswith(".mp3"):  # Ensure it's an audio file
        file_path = os.path.join(source_folder, filename)
        
        # Check if file is female
        if filename.startswith("female_"):
            shutil.move(file_path, destination_folder)
        
        # Check if file is male and in the allowed age groups
        elif filename.startswith("male_"):
            for age in allowed_male_ages:
                if f"_{age}_" in filename:
                    shutil.move(file_path, destination_folder)
                    break  # Stop checking once a match is found

print("File transfer complete!")
