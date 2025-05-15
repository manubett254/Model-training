import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
base_dir = "C:/EEE/en/en/finalSortedFiles"
csv_file = f"{base_dir}/male_female_features.csv"
df = pd.read_csv(csv_file)

# Drop the 'filename' column (not useful for correlation)
df = df.drop(columns=["filename"])

# Compute correlation with gender
correlation_matrix = df.corr()

# Get correlation values for gender (sorted by importance)
gender_corr = correlation_matrix["gender"].abs().sort_values(ascending=False)
print("🔍 Feature Correlation with Gender:\n", gender_corr)

# Keep only features with |r| > 0.3
selected_features = gender_corr[gender_corr > 0.3].index.tolist()

print("\n✅ Keeping Features with |r| > 0.3:", selected_features)

# Save the filtered dataset
df_filtered = df[selected_features]  # Keep only useful features
filtered_csv_path = f"{base_dir}/filtered_features.csv"
df_filtered.to_csv(filtered_csv_path, index=False)
print(f"✅ Refined dataset saved: {filtered_csv_path}")

# Visualize Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[selected_features].corr(), cmap="coolwarm", annot=True)
plt.title("Filtered Feature Correlation Heatmap")
plt.show()
