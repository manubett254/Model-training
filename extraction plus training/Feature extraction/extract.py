import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define dataset path and output CSV path
DATASET_PATH = r"C:\EEE\Big dataset\clip\finalDataset"
CSV_OUTPUT_PATH = r"C:\EEE\Big dataset\Feature extraction\features.csv"

# Function to extract audio features
def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=16000)  # Ensure 16kHz sample rate

        # Trim or pad audio to exactly 5 seconds (right padding only)
        target_length = sr * 5
        if len(y) > target_length:
            start_sample = np.random.randint(0, len(y) - target_length)
            y = y[start_sample:start_sample + target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCCs (13 coefficients)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chroma Features
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral Contrast
        zcr = librosa.feature.zero_crossing_rate(y)  # Zero-Crossing Rate (ZCR)
        rms = librosa.feature.rms(y=y)  # Root Mean Square (RMS) Energy
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Spectral Centroid
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # Spectral Bandwidth
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Spectral Rolloff
        hnr = librosa.effects.harmonic(y)  # Harmonic-to-Noise Ratio (HNR)
        pitches, _ = librosa.piptrack(y=y, sr=sr)  # Pitch/Fundamental Frequency (F0)

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
        print(f"Error processing {file_path}: {e}")
        return None

# Extract features from all files
feature_data = []
file_list = os.listdir(DATASET_PATH)

for file_name in tqdm(file_list, desc="Processing Audio Files"):
    file_path = os.path.join(DATASET_PATH, file_name)
    
    # Extract label info (gender & age) from filename
    parts = file_name.split("_")
    if len(parts) < 3:
        continue  # Skip files with unexpected naming format
    gender = parts[0]  # 'male' or 'female'
    age_group = parts[1]  # 'twenties', 'thirties', etc.
    client_id = parts[-1].split('.')[0]  # Extract client ID
    
    # Extract features
    features = extract_features(file_path)
    if features:
        feature_data.append([client_id] + features + [age_group, gender])

# Define column names
mfcc_cols = [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)]
chroma_cols = [f"chroma_{i}" for i in range(12)] + [f"chroma_std_{i}" for i in range(12)]
spec_contrast_cols = [f"spec_contrast_{i}" for i in range(7)] + [f"spec_contrast_std_{i}" for i in range(7)]
other_features = ["zcr", "zcr_std", "rms", "rms_std", "centroid", "centroid_std", "bandwidth", "bandwidth_std", "rolloff", "rolloff_std", "hnr", "hnr_std", "pitch", "pitch_std"]
columns = ["client_id"] + mfcc_cols + chroma_cols + spec_contrast_cols + other_features + ["age_group", "gender"]

# Convert to DataFrame and save
if feature_data:
    df = pd.DataFrame(feature_data, columns=columns)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print("Feature extraction complete. CSV saved at:", CSV_OUTPUT_PATH)
else:
    print("No valid features extracted.")
