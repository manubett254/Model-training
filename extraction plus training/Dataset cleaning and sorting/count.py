import os

# Set the path to your final dataset folder
final_dataset_path = r"C:\EEE\Big dataset\clip\finalDataset"

# Initialize counters
male_count = 0
female_count = 0

# Iterate through the files in the folder
for filename in os.listdir(final_dataset_path):
    if filename.lower().startswith("male"):
        male_count += 1
    elif filename.lower().startswith("female"):
        female_count += 1

# Print results
print(f"Male files: {male_count}")
print(f"Female files: {female_count}")
