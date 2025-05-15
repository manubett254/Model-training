import os
import librosa
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Define dataset paths
DATASET_PATH = r"C:\EEE\test dataset\VoxCeleb_gender"  
CSV_OUTPUT_PATH = r"C:\EEE\Big dataset\Feature extraction\VoxCelebfeatures.csv"

# Set subfolder names
MALE_FOLDER = os.path.join(DATASET_PATH, "males")
FEMALE_FOLDER = os.path.join(DATASET_PATH, "females")

# Target number of files per gender
TARGET_SAMPLES = 2300

# Ensure the dataset structure is correct
if not os.path.exists(MALE_FOLDER) or not os.path.exists(FEMALE_FOLDER):
    raise FileNotFoundError("Male or Female subfolder not found! Check dataset path.")

# Function to extract audio features
def extract_features(file_path):
    try:
        # Load audio file (16kHz sample rate)
        y, sr = librosa.load(file_path, sr=16000)

        # Trim or pad to exactly 5 seconds
        target_length = sr * 5
        if len(y) > target_length:
            start_sample = np.random.randint(0, len(y) - target_length)
            y = y[start_sample:start_sample + target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        hnr = librosa.effects.harmonic(y)
        pitches, _ = librosa.piptrack(y=y, sr=sr)

        # Aggregate features (mean + std for each feature)
        features = {
            "mfcc": np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]),
            "chroma": np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)]),
            "spectral_contrast": np.concatenate([np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1)]),
            "zcr": [np.mean(zcr), np.std(zcr)],
            "rms": [np.mean(rms), np.std(rms)],
            "centroid": [np.mean(centroid), np.std(centroid)],
            "bandwidth": [np.mean(bandwidth), np.std(bandwidth)],
            "rolloff": [np.mean(rolloff), np.std(rolloff)],
            "hnr": [np.mean(hnr), np.std(hnr)],
            "pitch": [np.mean(pitches), np.std(pitches)]
        }

        # Flatten all feature arrays into a single list
        feature_vector = []
        for key, value in features.items():
            feature_vector.extend(value)

        return feature_vector
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Function to process files from a given folder
def process_files(folder_path, gender_label, sample_size):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".m4a")]
    selected_files = random.sample(all_files, min(sample_size, len(all_files)))  # Select up to sample_size files

    feature_data = []
    for file_path in tqdm(selected_files, desc=f"Processing {gender_label} audio"):
        features = extract_features(file_path)
        if features:
            feature_data.append([gender_label] + features)  # Gender label first

    return feature_data

# Process male and female files
male_features = process_files(MALE_FOLDER, "male", TARGET_SAMPLES)
female_features = process_files(FEMALE_FOLDER, "female", TARGET_SAMPLES)

# Combine data
feature_data = male_features + female_features
random.shuffle(feature_data)  # Mix male and female samples

# Define column names
mfcc_cols = [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)]
chroma_cols = [f"chroma_{i}" for i in range(12)] + [f"chroma_std_{i}" for i in range(12)]
spec_contrast_cols = [f"spec_contrast_{i}" for i in range(7)] + [f"spec_contrast_std_{i}" for i in range(7)]
other_features = ["zcr", "zcr_std", "rms", "rms_std", "centroid", "centroid_std", "bandwidth", "bandwidth_std", "rolloff", "rolloff_std", "hnr", "hnr_std", "pitch", "pitch_std"]
columns = ["gender"] + mfcc_cols + chroma_cols + spec_contrast_cols + other_features

# Convert to DataFrame and save
if feature_data:
    df = pd.DataFrame(feature_data, columns=columns)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"✅ Feature extraction complete. CSV saved at: {CSV_OUTPUT_PATH}")
else:
    print("❌ No valid features extracted.")
