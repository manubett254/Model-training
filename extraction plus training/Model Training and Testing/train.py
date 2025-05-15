import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("features.csv")

# Convert categorical labels to numeric
df["gender"] = df["gender"].map({"male": 0, "female": 1})  # Ensure binary classification

# Identify feature columns
feature_columns = [col for col in df.columns if col not in ["gender", "age_group"]]  # Exclude non-feature columns

# Drop non-numeric columns
X = df[feature_columns]
y = df["gender"]

# Print selected features
print(f"🟢 Training Features ({len(X.columns)}):", list(X.columns))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train, evaluate, and save models
def train_model(model, model_name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Print results
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ {model_name} Accuracy: {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and scaler
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"models/scaler_{model_name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"💾 Saved: models/{model_name}.pkl & models/scaler_{model_name}.pkl")

# Train models
train_model(RandomForestClassifier(n_estimators=100, random_state=42), "gender_model_rf")
train_model(SVC(kernel="linear", probability=True), "gender_model_svm")
train_model(LogisticRegression(max_iter=1000), "gender_model_lr")

# Save feature list for consistency in prediction
with open("models/feature_list.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("✅ Feature list saved to models/feature_list.pkl")
