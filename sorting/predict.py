import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa

# Define paths
base_dir = "C:/EEE/en/en/finalSortedFiles"
model_path = f"{base_dir}/logistic_regression_model.pkl"
csv_file = f"{base_dir}/filtered_features.csv"  # For getting feature scaling parameters

# Load the saved model
model = joblib.load(model_path)
print("✅ Model Loaded Successfully!")


# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # Load audio at 16kHz

    # Extract features (same as training)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # 40 MFCCs
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # Chroma features
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral contrast
    zcr = librosa.feature.zero_crossing_rate(y)  # Zero-crossing rate
    rms_energy = librosa.feature.rms(y=y)  # RMS Energy
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Spectral centroid
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # Spectral bandwidth
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Spectral roll-off

    # Convert features to 1D arrays (take mean across time)
    features = np.hstack([
        np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spec_contrast, axis=1),
        np.mean(zcr), np.mean(rms_energy), np.mean(spec_centroid),
        np.mean(spec_bandwidth), np.mean(spec_rolloff)
    ])

    return features

# Path to a new audio file
new_audio_file = "C:/EEE/en/en/finalSortedFile/sother/adult/other_adult_1d3b51e0050fbe2c29507fda3797ab28e9c522c9542597c4706ad953d1eb54f29e456ebdcb315a21f415bef4b8f1585c1d7fb45784748ece8b480f4c67b43f8f.mp3C:\EEE\sorting\other_adult_1cde41e74e14d85e89b20f5e94d57b668b1743ea21fa198260fd9d6e8270628e49c95ebcb143d84407bcbbb02e6da2d7d67117844b5fcbce06bfbe497fdf016d.mp3"  # Change to actual file

# Extract features
new_features = extract_features(new_audio_file)

# Scale features using previously fitted scaler
new_features_scaled = scaler.transform([new_features])

# Predict gender (0 = male, 1 = female)
prediction = model.predict(new_features_scaled)[0]

# Output result
gender = "Female" if prediction == 1 else "Male"
print(f"🎯 Predicted Gender: {gender}")
