import pandas as pd
import os
import shutil

# 🔹 Step 1: Load the metadata file (Use 'train.tsv' or 'validated.tsv')
df = pd.read_csv("validated.tsv", sep="\t")

# 🔹 Step 2: Keep only rows where 'age' and 'gender' are not empty
df_filtered = df.dropna(subset=['age', 'gender'])

# 🔹 Step 3: Get the list of valid audio file names
valid_files = set(df_filtered["path"])  # 'path' column contains audio file names

# 🔹 Step 4: Define audio directory paths
audio_dir = "C:/EEE/Big dataset/clips"  # Change this to the actual path of your audio files
output_dir = "C:/EEE/Big dataset/clip/validFiles"  # Where valid files will be moved

# 🔹 Step 5: Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 🔹 Step 6: Move only valid audio files
for filename in os.listdir(audio_dir):
    if filename in valid_files:
        shutil.move(os.path.join(audio_dir, filename), os.path.join(output_dir, filename))

print(f"✅ Filtering complete! Kept {len(valid_files)} files in '{output_dir}'.")
