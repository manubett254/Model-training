import os
import librosa
import numpy as np
import pandas as pd

# Define paths
base_dir = "C:/EEE/en/en/finalSortedFiles"

# Folders to process
folders_to_process = {
    "male_adult": os.path.join(base_dir, "male", "adult"),
    "male_teen": os.path.join(base_dir, "male", "teen"),
    "female_adult": os.path.join(base_dir, "female", "adult"),
    "female_teen": os.path.join(base_dir, "female", "teen"),
}

# Function to extract audio features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)  # Load audio with a fixed sample rate
        
        # Extract Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # 40 MFCCs
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chroma features
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral contrast
        zcr = librosa.feature.zero_crossing_rate(y)  # Zero-crossing rate
        rms_energy = librosa.feature.rms(y=y)  # RMS Energy
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Spectral centroid
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # Spectral bandwidth
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Spectral roll-off

        # Convert to 1D arrays (taking the mean across time frames)
        mfccs_mean = np.mean(mfccs, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        zcr_mean = np.mean(zcr)
        rms_mean = np.mean(rms_energy)
        spec_centroid_mean = np.mean(spec_centroid)
        spec_bandwidth_mean = np.mean(spec_bandwidth)
        spec_rolloff_mean = np.mean(spec_rolloff)

        # Combine features into one array
        features = np.hstack([
            mfccs_mean, chroma_mean, spec_contrast_mean, 
            zcr_mean, rms_mean, spec_centroid_mean, 
            spec_bandwidth_mean, spec_rolloff_mean
        ])

        return features

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# Process each folder and save features
for category, folder_path in folders_to_process.items():
    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping {category}, folder not found: {folder_path}")
        continue

    # Get all MP3 files
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp3")])
    print(f"🔄 Processing {len(files)} files in {category}...")

    # List to store feature rows
    feature_list = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        if features is not None:
            feature_list.append([file] + features.tolist())  # Store filename + features

    # Convert to DataFrame
    columns = (["filename"] + 
               [f"mfcc_{i+1}" for i in range(40)] + 
               [f"chroma_{i+1}" for i in range(12)] + 
               [f"spec_contrast_{i+1}" for i in range(7)] + 
               ["zero_crossing_rate", "rms_energy", "spectral_centroid", 
                "spectral_bandwidth", "spectral_rolloff"])

    df = pd.DataFrame(feature_list, columns=columns)

    # Save to CSV
    csv_file_path = os.path.join(base_dir, f"{category}_features.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"✅ Saved features to {csv_file_path}")

print("🎵✅ Feature extraction complete!")
