import os

# Path to dataset
dataset_path = r"C:\EEE\Datasets\age dataset\Train"

# Define age groups
age_groups = {
    "Child": ["female_child", "male_child"],
    "Teens": ["female_teens", "male_teens"],
    "Middle-aged Adults": ["female_twenties", "female_thirties", "female_fourties",
                           "male_twenties", "male_thirties", "male_fourties"],
    "Older Adults": ["female_fifties", "female_sixties", "female_seventies", "female_eighties",
                     "male_fifties", "male_sixties", "male_seventies", "male_eighties"]
}

# Dictionary to store counts
age_group_counts = {group: 0 for group in age_groups}

# Count files for each age group
for group, folders in age_groups.items():
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            age_group_counts[group] += len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Print results
for group, count in age_group_counts.items():
    print(f"{group}: {count} files")
