import os
import joblib
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# Define paths for models and data
CLIPS_FOLDER = r"C:\EEE\BESD\ENGLISH\NEUTRAL"
SVM_MODEL_PATH = r"C:\EEE\Big dataset\Model Training and Testing\models\gender_model_svm.pkl"
LR_MODEL_PATH = r"C:\EEE\Big dataset\Model Training and Testing\models\gender_model_lr.pkl"
SCALER_PATH = r"C:\EEE\Big dataset\Model Training and Testing\models\scaler.pkl"
FEATURE_LIST_PATH = r"C:\EEE\Big dataset\Model Training and Testing\models\feature_list.pkl"

# Load models and scaler
with open(SVM_MODEL_PATH, "rb") as f:
    svm_model = joblib.load(f)
with open(LR_MODEL_PATH, "rb") as f:
    lr_model = joblib.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = joblib.load(f)
with open(FEATURE_LIST_PATH, "rb") as f:
    feature_list = joblib.load(f)

# Initialize gender counters and filename lists
svm_male_count = 0
svm_female_count = 0
lr_male_count = 0
lr_female_count = 0

svm_male_files = []
svm_female_files = []
lr_male_files = []
lr_female_files = []

def extract_features(file_path):
    """Extracts features from an audio file."""
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

        # Flatten feature arrays into a single list
        feature_vector = []
        for value in features.values():
            feature_vector.extend(value)

        return feature_vector
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Process audio files
for file_name in os.listdir(CLIPS_FOLDER):
    if file_name.endswith((".wav", ".mp3", ".m4a")):
        file_path = os.path.join(CLIPS_FOLDER, file_name)
        print(f"🔍 Processing {file_name}...")

        features = extract_features(file_path)
        if features:
            df_features = pd.DataFrame([features], columns=feature_list)
            scaled_features = scaler.transform(df_features)

            # Predictions
            svm_prediction = svm_model.predict(scaled_features)[0]
            lr_prediction = lr_model.predict(scaled_features)[0]

            svm_gender = "female" if svm_prediction == 1 else "male"
            lr_gender = "female" if lr_prediction == 1 else "male"

            # Update counters and store filenames
            if svm_gender == "female":
                svm_female_count += 1
                svm_female_files.append(file_name)
            else:
                svm_male_count += 1
                svm_male_files.append(file_name)

            if lr_gender == "female":
                lr_female_count += 1
                lr_female_files.append(file_name)
            else:
                lr_male_count += 1
                lr_male_files.append(file_name)

            # Print results with counts
            print(f"🎤 {file_name} → SVM: {svm_gender} | LR: {lr_gender}")
            print(f"📊 SVM - Female: {svm_female_count} | Male: {svm_male_count}")
            print(f"📊 LR  - Female: {lr_female_count} | Male: {lr_male_count}\n")
        else:
            print(f"❌ Failed to extract features from {file_name}\n")

# Print final summary with filenames
print("\n🔹 FINAL PREDICTIONS SUMMARY 🔹")
print("\nSVM Model Predictions:")
print(f"📁 Files Classified as Male ({svm_male_count}):")
print("\n".join(svm_male_files) if svm_male_files else "None")
print(f"\n📁 Files Classified as Female ({svm_female_count}):")
print("\n".join(svm_female_files) if svm_female_files else "None")

print("\nLR Model Predictions:")
print(f"📁 Files Classified as Male ({lr_male_count}):")
print("\n".join(lr_male_files) if lr_male_files else "None")
print(f"\n📁 Files Classified as Female ({lr_female_count}):")
print("\n".join(lr_female_files) if lr_female_files else "None")
