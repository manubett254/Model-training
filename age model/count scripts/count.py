import os

# Path to dataset
dataset_path = r"C:\EEE\Datasets\age dataset\Train"

# Get all folders in the dataset directory
folders = sorted(os.listdir(dataset_path))

# Dictionary to store file counts
file_counts = {}

# Loop through each folder and count files
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        file_counts[folder] = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Print results
for folder, count in file_counts.items():
    print(f"{folder}: {count} files")
