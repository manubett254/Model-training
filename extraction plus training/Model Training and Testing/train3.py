import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Handling class imbalance

# Define the save directory
save_dir = r"C:\EEE\Big dataset\Model Training and Testing\kidMOdels"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Load dataset
df = pd.read_csv("Gender_CSV.csv")

# Ensure no NaN values in labels
df = df.dropna(subset=["gender"])

# Convert gender to integer (0 = Male, 1 = Female)
df["gender"] = df["gender"].astype(int)

# Identify feature columns
feature_columns = [col for col in df.columns if col not in ["gender", "age_group"]]

# Select features and labels
X = df[feature_columns]
y = df["gender"]

# Check for class imbalance (Kids might be underrepresented)
kid_count = df[df["age_group"] == "teens"].shape[0]
adult_count = df[df["age_group"] != "teens"].shape[0]
print(f"👶 Kids: {kid_count} | 🏋 Adults: {adult_count}")

# Apply SMOTE if Kids are underrepresented
if kid_count < 0.3 * adult_count:  # If kids are less than 30% of adults
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, y = smote.fit_resample(X, y)
    print("🔄 Applied SMOTE to balance kid samples")

# Print selected features
print(f"🟢 Training Features ({len(X.columns)}):", list(X.columns))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---------------------- SVM Hyperparameter Tuning ---------------------- #
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.1, 1, 10]
}

svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring='accuracy', verbose=2)
svm_grid.fit(X_train_scaled, y_train)

# Best SVM Model
best_svm = svm_grid.best_estimator_
print(f"✅ Best SVM Parameters: {svm_grid.best_params_}")

# Evaluate SVM
svm_pred = best_svm.predict(X_test_scaled)
print("📊 SVM Classification Report:\n", classification_report(y_test, svm_pred))

# Save SVM model
with open(os.path.join(save_dir, "gender_model_svm.pkl"), "wb") as f:
    pickle.dump(best_svm, f)

# ---------------------- Logistic Regression Hyperparameter Tuning ---------------------- #
lr_params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l1', 'l2']
}

lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='accuracy', verbose=2)
lr_grid.fit(X_train_scaled, y_train)

# Best Logistic Regression Model
best_lr = lr_grid.best_estimator_
print(f"✅ Best Logistic Regression Parameters: {lr_grid.best_params_}")

# Evaluate LR
lr_pred = best_lr.predict(X_test_scaled)
print("📊 Logistic Regression Classification Report:\n", classification_report(y_test, lr_pred))

# Save LR model
with open(os.path.join(save_dir, "gender_model_lr.pkl"), "wb") as f:
    pickle.dump(best_lr, f)

# ---------------------- Feature Importance Analysis ---------------------- #
print("\n🔍 Feature Importance Analysis:")
svm_feature_weights = abs(best_svm.coef_[0]) if best_svm.kernel == 'linear' else None
lr_feature_weights = abs(best_lr.coef_[0])

if svm_feature_weights is not None:
    print("📊 SVM Feature Weights (Linear Kernel):")
    for feature, weight in sorted(zip(feature_columns, svm_feature_weights), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {weight:.4f}")

print("\n📊 Logistic Regression Feature Weights:")
for feature, weight in sorted(zip(feature_columns, lr_feature_weights), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {weight:.4f}")

# ---------------------- Compare Models ---------------------- #
svm_acc = accuracy_score(y_test, svm_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"🎯 SVM Accuracy: {svm_acc:.4f}")
print(f"🎯 Logistic Regression Accuracy: {lr_acc:.4f}")

if svm_acc > lr_acc:
    print("🚀 SVM is the better model!")
else:
    print("🚀 Logistic Regression is the better model!")

# Save feature list for consistency in prediction
with open(os.path.join(save_dir, "feature_list.pkl"), "wb") as f:
    pickle.dump(feature_columns, f)

print(f"✅ Feature list saved to {save_dir}")
