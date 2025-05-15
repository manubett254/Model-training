import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load your dataset (update the path as needed)
df = pd.read_csv('/content/ageCSV final.csv')  # Must have a 'label' column with 'child', 'teen', 'adult'

# ------------------------------
# STEP 1: Binary Classifier – Child vs Non-Child
# ------------------------------
# Encode binary labels
label_encoder_step1 = LabelEncoder()
df['binary_label'] = df['age_group'].apply(lambda x: 'child' if x == 'child' else 'non-child')
y_step1 = label_encoder_step1.fit_transform(df['binary_label'])

X_step1 = df.drop(columns=['age_group', 'binary_label'])

# Scale features
scaler_step1 = StandardScaler()
X_step1_scaled = scaler_step1.fit_transform(X_step1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_step1_scaled, y_step1, test_size=0.2, random_state=42)

model_step1 = SVC(kernel='rbf', probability=True, class_weight='balanced')
model_step1.fit(X_train1, y_train1)

print("Step 1: Child vs Non-Child")
y_pred1 = model_step1.predict(X_test1)
print(classification_report(y_test1, y_pred1))
print(confusion_matrix(y_test1, y_pred1))

# ------------------------------
# STEP 2: Teen vs Adult (only for non-child)
# ------------------------------
non_child_df = df[df['binary_label'] == 'non-child'].copy()

X_step2 = non_child_df.drop(columns=['age_group', 'binary_label'])
y_step2 = non_child_df['age_group'].replace('teens', 'teen')

# Encode labels for Step 2
label_encoder_step2 = LabelEncoder()
y_step2_encoded = label_encoder_step2.fit_transform(y_step2)

# Oversample 'teen' using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
print("Before SMOTE:\n", pd.Series(y_step2_encoded).value_counts())
X_resampled, y_resampled = smote.fit_resample(X_step2, y_step2_encoded)
print("After SMOTE:\n", pd.Series(y_resampled).value_counts())

# Scale features
scaler_step2 = StandardScaler()
X_resampled_scaled = scaler_step2.fit_transform(X_resampled)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

# Train Step 2 model
model_step2 = RandomForestClassifier(class_weight='balanced', n_estimators=100)
model_step2.fit(X_train2, y_train2)

print("Step 2: Teen vs Adult")
y_pred2 = model_step2.predict(X_test2)
print(classification_report(y_test2, y_pred2))
print(confusion_matrix(y_test2, y_pred2))

# ------------------------------
# Save the models, scalers, and encoders
# ------------------------------
joblib.dump(label_encoder_step1, 'label_encoder_step1.joblib')
joblib.dump(scaler_step1, 'scaler_step1.joblib')
joblib.dump(model_step1, 'model_step1.joblib')

joblib.dump(label_encoder_step2, 'label_encoder_step2.joblib')
joblib.dump(scaler_step2, 'scaler_step2.joblib')
joblib.dump(model_step2, 'model_step2.joblib')

print("All models, scalers, and encoders saved successfully!")