import pandas as pd
import os
import random

# Define paths
base_dir = "C:/EEE/en/en/finalSortedFiles"

# File paths for individual feature CSVs
csv_files = {
    "male_adult": os.path.join(base_dir, "male_adult_features.csv"),
    "male_teen": os.path.join(base_dir, "male_teen_features.csv"),
    "female_adult": os.path.join(base_dir, "female_adult_features.csv"),
    "female_teen": os.path.join(base_dir, "female_teen_features.csv"),
}

# Load CSV files into DataFrames
dfs = {name: pd.read_csv(path) for name, path in csv_files.items() if os.path.exists(path)}

# Merge male datasets
df_male = pd.concat([dfs["male_adult"], dfs["male_teen"]], ignore_index=True)
df_male["gender"] = 0  # 0 = Male

# Merge female datasets
df_female = pd.concat([dfs["female_adult"], dfs["female_teen"]], ignore_index=True)
df_female["gender"] = 1  # 1 = Female

# Get number of female samples (limit for males)
num_females = len(df_female)

# Randomly select an equal number of male samples
df_male = df_male.sample(n=num_females, random_state=42).reset_index(drop=True)

# Interleave male and female rows (male → female → male → female)
combined_list = []
male_index, female_index = 0, 0

while male_index < len(df_male) and female_index < len(df_female):
    combined_list.append(df_male.iloc[male_index])
    combined_list.append(df_female.iloc[female_index])
    male_index += 1
    female_index += 1

# Convert back to DataFrame
df_combined = pd.DataFrame(combined_list)

# Save to CSV
output_csv_path = os.path.join(base_dir, "male_female_features.csv")
df_combined.to_csv(output_csv_path, index=False)

print(f"✅ Combined dataset saved: {output_csv_path}")
print(f"🔹 Total Samples: {len(df_combined)} (Male: {len(df_male)}, Female: {len(df_female)})")
