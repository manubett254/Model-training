import os
import pandas as pd
import shutil

# Define paths
old_audio_folder = 'C:/EEE/en/en/sortedFiles'  # Folder with fifties, forties, etc.
csv_file_path = 'C:/EEE/en/en/validated_distinct.csv'  # Metadata file
new_audio_folder = 'C:/EEE/en/en/finalSortedFiles'  # Target folder

# Read CSV and filter valid rows
df = pd.read_csv(csv_file_path)
df_filtered = df.dropna(subset=['path', 'age', 'gender'])  # Remove missing values

# Define correct folder mapping
folder_mapping = {
    "teens": "teen",
    "twenties": "adult",
    "thirties": "adult",
    "forties": "adult",
    "fifties": "adult",
    "sixties": "adult",
}

# Debug: Print available folders
print("Existing male folders:", os.listdir(os.path.join(old_audio_folder, "male")))
print("Existing female folders:", os.listdir(os.path.join(old_audio_folder, "female")))

# Track files moved
files_processed = 0

# Iterate over files and move them
for index, row in df_filtered.iterrows():
    original_name = str(row['path']).strip().lower() + ".mp3"  # Original filename
    gender = str(row['gender']).lower()

    # Find the folder containing this file
    found_folder = None
    for folder in os.listdir(os.path.join(old_audio_folder, gender)):  # Check ALL folders
        if folder.lower() in folder_mapping.keys():  
            test_path = os.path.join(old_audio_folder, gender, folder, original_name)
            if os.path.exists(test_path):  # ✅ Check if the file exists in this folder
                found_folder = folder
                break  

    # Skip if no matching folder was found
    if found_folder is None:
        print(f"⚠️ No matching folder found for {original_name}")
        continue  

    # Get mapped age group
    age_group = folder_mapping[found_folder.lower()]
    
    # Define old and new folder paths
    old_folder_path = os.path.join(old_audio_folder, gender, found_folder)
    destination_folder = os.path.join(new_audio_folder, gender, age_group)
    os.makedirs(destination_folder, exist_ok=True)  # Ensure the new folder exists

    original_path = os.path.join(old_folder_path, original_name)
    new_name = f"{gender}_{age_group}_{original_name}"  # New filename format
    new_path = os.path.join(destination_folder, new_name)

    # Debug: Print expected paths
    print(f"Looking for: {original_path}")

    # Move and rename file
    if os.path.exists(original_path):
        shutil.move(original_path, new_path)
        files_processed += 1
    else:
        print(f"⚠️ File not found: {original_path}")  # Debug missing files

print(f"✅ {files_processed} files have been sorted into new folders!")
