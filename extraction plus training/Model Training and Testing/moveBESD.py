import os
import shutil
import random

# Paths
BESD_FOLDER = r"C:\EEE\BESD"  # Main BESD dataset
OUTPUT_FOLDER = r"C:\EEE\BESD_Split"  # New structured dataset folder

TRAIN_RATIO = 0.8  # 80% training, 20% testing

# Create train and test folders
train_folder = os.path.join(OUTPUT_FOLDER, "train")
test_folder = os.path.join(OUTPUT_FOLDER, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Collect all BESD audio files
all_files = []
for lang in ["ENGLISH", "TELUGU"]:
    lang_path = os.path.join(BESD_FOLDER, lang)
    for emotion in os.listdir(lang_path):  # Loop through emotions
        emotion_path = os.path.join(lang_path, emotion)
        if os.path.isdir(emotion_path):
            for file in os.listdir(emotion_path):
                if file.endswith(".wav"):
                    all_files.append(os.path.join(emotion_path, file))

# Shuffle and split dataset
random.shuffle(all_files)
split_idx = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

# Function to copy and rename files
def copy_and_rename(files, dest_folder):
    for idx, file_path in enumerate(files):
        # Extract gender & age from filename (Example: 1.EF_12_happy_1.wav → gender=female, age=12)
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        
        if len(parts) < 3:
            print(f"⚠️ Skipping {filename} (Invalid Format)")
            continue
        
        gender_part = parts[0]  # e.g., "EF" or "EM"
        age = parts[1]  # e.g., "12"
        
        gender = "male" if "M" in gender_part else "female"
        age_group = "child"  # Since all BESD speakers are 6-12

        # New filename format: "gender_agegroup_serial.wav"
        new_filename = f"{gender}_{age_group}_{idx + 1}.wav"
        target_path = os.path.join(dest_folder, new_filename)

        # Copy file
        shutil.copy(file_path, target_path)
        print(f"✅ Copied {filename} → {new_filename} into {dest_folder}")

# Copy training and testing data
print("\n📂 Organizing TRAINING data...")
copy_and_rename(train_files, train_folder)

print("\n📂 Organizing TESTING data...")
copy_and_rename(test_files, test_folder)

print("\n🚀 BESD Dataset Successfully Organized!")
print(f"🔹 Training samples stored in: {train_folder}")
print(f"🔹 Testing samples stored in: {test_folder}")
