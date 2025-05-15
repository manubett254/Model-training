import pandas as pd

csv_path = r"C:\EEE\Big dataset\validated_cleaned_distinct.csv"
df = pd.read_csv(csv_path)

# Count occurrences of each gender
gender_counts = df["gender"].value_counts()

print(gender_counts)
