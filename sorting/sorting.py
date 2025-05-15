import os
import shutil
import pandas as pd

# Paths to the folder with audio clips and the CSV file
audio_folder_path = 'C:/EEE/en/en/clips'
csv_file_path = 'c:/EEE/en/en/validated_distinct.csv'
output_folder_path = 'C:/EEE/en/en/validFiles'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Get the list of audio clip names from the 'paths' column
audio_clip_names = df['path'].tolist()

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over the files in the audio folder
for file_name in os.listdir(audio_folder_path):
    # Check if the file name (without extension) is in the list of audio clip names
    if os.path.splitext(file_name)[0] in audio_clip_names:
        # Copy the file to the output folder
        shutil.copy(os.path.join(audio_folder_path, file_name), output_folder_path)

print("Audio clips have been sorted and copied to the output folder.")