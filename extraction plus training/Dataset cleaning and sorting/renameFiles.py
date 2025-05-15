import os
import shutil
import pandas as pd

# Load the CSV file
csv_path = r"C:\EEE\Big dataset\validated_cleaned_distinct.csv"
df = pd.read_csv(csv_path)

# Define source and destination directories
src_folder = r"C:\EEE\Big dataset\clip\validFiles"
dest_folder = r"C:\EEE\Big dataset\clip\renamedFiles"

# Ensure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

# Sort dataframe by client_id for consistency
df = df.sort_values(by="client_id")

# Generate new sequential IDs
df["new_id"] = df.reset_index().index + 1  # Generates 0001, 0002, ...

# Process files
copied_files = 0
missing_files = 0

for _, row in df.iterrows():
    file_name = row["path"].strip() + ".mp3"  # Original filename
    gender = "male" if row["gender"] == 0 else "female"  # Gender label
    age_category = row["age"].strip().lower()  # Use existing age category
    new_id = f"{row['new_id']:04d}"  # Format ID to 4 digits (0001, 0002)

    new_filename = f"{gender}_{age_category}_{new_id}.mp3"  # New filename
    src_path = os.path.join(src_folder, file_name)
    dest_path = os.path.join(dest_folder, new_filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        copied_files += 1
    else:
        print(f"🚨 File not found: {src_path}")
        missing_files += 1

print(f"✅ Done! {copied_files} files renamed and copied. {missing_files} missing files.")
