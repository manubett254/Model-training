import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define the save directory
save_dir = "/content/Voice analyzer/Models"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Load dataset
df = pd.read_csv("/content/Voice analyzer/Gender_CSV.csv")

# Ensure no NaN values in labels
df = df.dropna(subset=["gender"])

# Convert gender to integer (0 = Male, 1 = Female)
df["gender"] = df["gender"].astype(int)

# Identify feature columns
feature_columns = [col for col in df.columns if col != "gender"]

# Select features and labels
X = df[feature_columns]
y = df["gender"]

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

# SVM Feature Importance (Only for Linear Kernel)
if best_svm.kernel == 'linear':
    svm_feature_weights = abs(best_svm.coef_[0])
    print("\n📊 Top 10 SVM Feature Weights (Linear Kernel):")
    sorted_features_svm = sorted(zip(feature_columns, svm_feature_weights), key=lambda x: x[1], reverse=True)
    for feature, weight in sorted_features_svm[:10]:  # Top 10 features
        print(f"{feature}: {weight:.4f}")

# Logistic Regression Feature Importance
lr_feature_weights = abs(best_lr.coef_[0])
print("\n📊 Top 10 Logistic Regression Feature Weights:")
sorted_features_lr = sorted(zip(feature_columns, lr_feature_weights), key=lambda x: x[1], reverse=True)
for feature, weight in sorted_features_lr[:10]:  # Top 10 features
    print(f"{feature}: {weight:.4f}")

# ---------------------- Compare Models ---------------------- #
svm_acc = accuracy_score(y_test, svm_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"\n🎯 SVM Accuracy: {svm_acc:.4f}")
print(f"🎯 Logistic Regression Accuracy: {lr_acc:.4f}")

if svm_acc > lr_acc:
    print("🚀 SVM is the better model!")
else:
    print("🚀 Logistic Regression is the better model!")

# Save feature list for consistency in prediction
with open(os.path.join(save_dir, "feature_list.pkl"), "wb") as f:
    pickle.dump(feature_columns, f)

print(f"✅ Feature list saved to {save_dir}")
