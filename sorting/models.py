import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load the dataset
base_dir = "C:/EEE/en/en/finalSortedFiles"
csv_file = f"{base_dir}/filtered_features.csv"
df = pd.read_csv(csv_file)

# Separate features and labels
X = df.drop(columns=["gender"])  # Features
y = df["gender"]  # Labels (0 = male, 1 = female)

# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (important for SVM & Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC(kernel="linear"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)  # Train
    y_pred = model.predict(X_test)  # Predict
    acc = accuracy_score(y_test, y_pred)  # Evaluate
    results[name] = acc
    print(f"✅ {name} Accuracy: {acc:.4f}")

# Find the best model
best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with Accuracy: {results[best_model]:.4f}")
# Import library
from sklearn.linear_model import LogisticRegression

# Train with hyperparameter tuning
best_model = LogisticRegression(C=0.8, solver="lbfgs", max_iter=1000)  
best_model.fit(X_train, y_train)

# Predict & evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Tuned Logistic Regression Accuracy: {accuracy:.4f}")
import joblib

# Save the model
model_path = "C:/EEE/en/en/finalSortedFiles/logistic_regression_model.pkl"
joblib.dump(best_model, model_path)
print(f"✅ Model saved at: {model_path}")

# Load the model later
loaded_model = joblib.load(model_path)
