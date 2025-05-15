import os

# Define the sorted files directory
sorted_folder = "C:/EEE/en/en/finalSortedFiles"

# Define the subfolders to count files in
folders_to_check = {
    "Male Adult": os.path.join(sorted_folder, "male", "adult"),
    "Male Teen": os.path.join(sorted_folder, "male", "teen"),
    "Female Adult": os.path.join(sorted_folder, "female", "adult"),
    "Female Teen": os.path.join(sorted_folder, "female", "teen")
}

# Count and print the number of files in each folder
for category, folder_path in folders_to_check.items():
    if os.path.exists(folder_path):  # Ensure the folder exists before counting
        num_files = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
        print(f"{category}: {num_files} files")
    else:
        print(f"⚠️ Folder not found: {category}")
