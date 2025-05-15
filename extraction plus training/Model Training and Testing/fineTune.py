import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 🔹 Load original and new datasets
original_data_path = "original_training_data.csv"
new_data_path = "VoxCelebfeatures.csv"

df_old = pd.read_csv(original_data_path)
df_new = pd.read_csv(new_data_path)

# 🔹 Ensure 'gender' column exists
for df, name in [(df_old, "original dataset"), (df_new, "new dataset")]:
    if "gender" not in df.columns:
        raise ValueError(f"⚠️ 'gender' column not found in {name}!")

# 🔹 Drop rows with missing gender labels
df_old = df_old.dropna(subset=["gender"])
df_new = df_new.dropna(subset=["gender"])

# 🔹 Convert gender labels to 0 (male) and 1 (female) if needed
for df in [df_old, df_new]:
    if df["gender"].dtype == object:
        df["gender"] = df["gender"].map({"male": 0, "female": 1}).astype(int)

# 🔹 Load feature list from previous training
with open("models/feature_list.pkl", "rb") as f:
    feature_columns = joblib.load(f)

# 🔹 Ensure both datasets have the same features
df_old = df_old[["gender"] + feature_columns]
df_new = df_new[["gender"] + feature_columns]

# 🔹 Merge datasets & shuffle
df_combined = pd.concat([df_old, df_new], ignore_index=True).sample(frac=1, random_state=42)

# 🔹 Extract features and labels
X = df_combined[feature_columns]
y = df_combined["gender"]

# 🔹 Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Train Logistic Regression
print("🔄 Training Logistic Regression from scratch...")
lr_model = LogisticRegression(solver="saga", max_iter=5000, tol=1e-4)
lr_model.fit(X_train_scaled, y_train)

# 🔹 Train SVM
print("🔄 Training SVM from scratch...")
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train_scaled, y_train)

# 🔹 Save models & scaler
joblib.dump(lr_model, "models/gender_model_lr_new.pkl")
joblib.dump(svm_model, "models/gender_model_svm_new.pkl")
joblib.dump(scaler, "models/scaler_gender_model_new.pkl")

print("🎯 Training complete! New models and scaler saved successfully.")
