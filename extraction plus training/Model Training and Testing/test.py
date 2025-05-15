import os
import pickle
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# Define paths
CLIPS_FOLDER = r"C:\EEE\Age Dataset\SortedFiles\Age_dataset"
MODEL_PATH = "models/gender_model_svm.pkl"
SCALER_PATH = "models/scaler_gender_model_svm.pkl"
FEATURE_LIST_PATH = "models/feature_list.pkl"

# Load the trained model and scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(FEATURE_LIST_PATH, "rb") as f:
    feature_list = pickle.load(f)

# Counters for gender predictions
male_count = 0
female_count = 0

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

# Load and test audio clips
for file_name in os.listdir(CLIPS_FOLDER):
    if file_name.endswith((".wav", ".mp3", ".m4a")):
        file_path = os.path.join(CLIPS_FOLDER, file_name)
        print(f"🔍 Processing {file_name}...")

        features = extract_features(file_path)
        if features:
            df_features = pd.DataFrame([features], columns=feature_list)
            scaled_features = scaler.transform(df_features)
            prediction = model.predict(scaled_features)[0]
            predicted_gender = "female" if prediction == 1 else "male"

            # Update counters
            if predicted_gender == "female":
                female_count += 1
            else:
                male_count += 1

            # Print results with counts
            print(f"🎤 {file_name} → Predicted Gender: {predicted_gender}")
            print(f"📊 Female Count: {female_count} | Male Count: {male_count}\n")
        else:
            print(f"❌ Failed to extract features from {file_name}\n")
