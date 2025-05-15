import os
import csv

# Define the main directory
base_dir = "C:/EEE/en/en/finalSortedFiles"

# Define the folders to process (excluding 'child' and 'others')
folders_to_process = {
    "male_adult": os.path.join(base_dir, "male", "adult"),
    "male_teen": os.path.join(base_dir, "male", "teen"),
    "female_adult": os.path.join(base_dir, "female", "adult"),
    "female_teen": os.path.join(base_dir, "female", "teen"),
}

# CSV file to store the mapping
csv_file_path = os.path.join(base_dir, "file_mapping.csv")

# Open the CSV file to write the mapping
with open(csv_file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Original Filename", "New Filename", "Category"])  # Header

    # Process each category folder
    for category, folder_path in folders_to_process.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ Skipping {category}, folder not found: {folder_path}")
            continue

        # Get all MP3 files and sort them
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp3")])

        print(f"🔄 Processing {len(files)} files in {category}...")

        # Rename each file with an indexed name
        for index, file in enumerate(files, start=1):
            new_filename = f"{category}_{index:04d}.mp3"  # Format as 0001, 0002...
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_filename)

            try:
                os.rename(old_path, new_path)  # Rename the file
                writer.writerow([file, new_filename, category])  # Store in CSV
            except Exception as e:
                print(f"⚠️ Error renaming {file}: {e}")

print(f"✅ Renaming complete! File mapping saved in {csv_file_path}")
