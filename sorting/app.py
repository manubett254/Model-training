import os
import joblib
import librosa
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Define paths
base_dir = "C:/EEE/en/en/finalSortedFiles"
model_path = f"{base_dir}/logistic_regression_model.pkl"
csv_file = f"{base_dir}/filtered_features.csv"  # For scaler

# Load trained model
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model Loaded Successfully!")
else:
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

# Load dataset & prepare scaler
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)  # Load CSV
    print("✅ CSV Loaded Successfully!")
else:
    raise FileNotFoundError(f"❌ CSV file not found: {csv_file}")

# Ensure 'df' contains necessary features
if "gender" in df.columns:
    X_train = df.drop(columns=["gender"])  # Features used for training
else:
    raise ValueError("❌ 'gender' column missing in CSV!")

# Fit the scaler using training data
scaler = StandardScaler()
scaler.fit(X_train)

# Function to extract features from an uploaded audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        # Extract features (same as training)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms_energy = librosa.feature.rms(y=y)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Convert to 1D array (take mean)
        features = np.hstack([
            np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spec_contrast, axis=1),
            np.mean(zcr), np.mean(rms_energy), np.mean(spec_centroid),
            np.mean(spec_bandwidth), np.mean(spec_rolloff)
        ])
        
        return features

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# API Route to Handle File Uploads
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp_audio.wav"
    file.save(file_path)

    try:
        # Extract features
        features = extract_features(file_path)

        if features is None:
            return jsonify({"error": "Failed to extract features"}), 500

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict gender
        prediction = model.predict(features_scaled)[0]
        gender = "Female" if prediction == 1 else "Male"

        # Return JSON response
        return jsonify({"gender": gender, "success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

# Run the API
if __name__ == "__main__":
    app.run(debug=True)
