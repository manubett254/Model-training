import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\EEE\Big dataset\Feature extraction\features.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Convert gender to categorical if not already
df['gender'] = df['gender'].astype(str)

# Plot gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df, palette='coolwarm')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("gender_distribution.png", dpi=300)  # Save the figure
plt.show()

# Feature distributions
feature_columns = df.columns[1:-2]  # Excluding client_id, gender, and age
df[feature_columns].hist(figsize=(15, 12), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.savefig("feature_distributions.png", dpi=300)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[feature_columns].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_correlation.png", dpi=300)
plt.show()
