import pandas as pd

# Load the datasets
features_path = "features.csv"
voxceleb_path = "VoxCelebfeatures.csv"

features_df = pd.read_csv(features_path)
voxceleb_df = pd.read_csv(voxceleb_path)

# Drop 'age_group' column from features.csv
if 'age_group' in features_df.columns:
    features_df = features_df.drop(columns=['age_group'])

# Ensure gender column is at the first position in both datasets
if features_df.columns[0] != 'gender':
    print("⚠️ Warning: 'gender' is not the first column in features.csv!")

gender_col = voxceleb_df['gender']  # Extract gender column
voxceleb_df = voxceleb_df.drop(columns=['gender'])  # Remove it from original place
voxceleb_df.insert(0, 'gender', gender_col)  # Insert at first position

# Ensure gender values are mapped consistently (0 for male, 1 for female)
def map_gender(value):
    if isinstance(value, str):
        return 0 if value.lower() == 'male' else 1  # Convert male to 0, female to 1
    return value  # If already numeric, return as is

features_df['gender'] = features_df['gender'].apply(map_gender)
voxceleb_df['gender'] = voxceleb_df['gender'].apply(map_gender)

# Merge datasets
merged_df = pd.concat([features_df, voxceleb_df], ignore_index=True)

# Save the merged dataset
merged_path = "merged_features.csv"
merged_df.to_csv(merged_path, index=False)
print(f"✅ Merged dataset saved as {merged_path}")
