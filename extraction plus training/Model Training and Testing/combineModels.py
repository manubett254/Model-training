import os
import pickle
import numpy as np
import librosa
import pandas as pd
from collections import Counter

# Define paths to models and scalers
MODEL_PATHS = {
    "svm": "models/gender_model_svm.pkl",
    "lr": "models/gender_model_lr.pkl",
    "rf": "models/gender_model_rf.pkl"
}
SCALER_PATHS = {
    "svm": "models/scaler_gender_model_svm.pkl",
    "lr": "models/scaler_gender_model_lr.pkl",
    "rf": "models/scaler_gender_model_rf.pkl"
}
FEATURE_LIST_PATH = "models/feature_list.pkl"

# Load models and scalers
models = {}
scalers = {}
for key in MODEL_PATHS:
    with open(MODEL_PATHS[key], "rb") as f:
        models[key] = pickle.load(f)
    with open(SCALER_PATHS[key], "rb") as f:
        scalers[key] = pickle.load(f)

# Load feature list
with open(FEATURE_LIST_PATH, "rb") as f:
    feature_list = pickle.load(f)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        target_length = sr * 5
        if len(y) > target_length:
            start_sample = np.random.randint(0, len(y) - target_length)
            y = y[start_sample:start_sample + target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')

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
        
        feature_vector = []
        for key, value in features.items():
            feature_vector.extend(value)
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def majority_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]

# Load and test audio clips
CLIPS_FOLDER = "C:/EEE/5.1/FEE 560/UI/New UI/voice-analyzer/clips/male/teen"  # Adjust as needed
male_count, female_count = 0, 0
for file_name in os.listdir(CLIPS_FOLDER):
    if file_name.endswith((".wav", ".mp3")):
        file_path = os.path.join(CLIPS_FOLDER, file_name)
        print(f"🔍 Processing {file_name}...")
        
        features = extract_features(file_path)
        if features:
            df_features = pd.DataFrame([features], columns=feature_list)
            
            predictions = []
            for key in models:
                scaled_features = scalers[key].transform(df_features)
                prediction = models[key].predict(scaled_features)[0]
                predictions.append(prediction)
            
            final_prediction = majority_vote(predictions)
            predicted_gender = "female" if final_prediction == 1 else "male"
            
            if predicted_gender == "male":
                male_count += 1
            else:
                female_count += 1
                
            print(f"🎤 {file_name} → Predicted Gender: {predicted_gender}")
            print(f"Current count → Male: {male_count}, Female: {female_count}\n")
        else:
            print(f"❌ Failed to extract features from {file_name}\n")
