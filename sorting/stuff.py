import pandas as pd

# Load the dataset
df = pd.read_csv("C:\EEE\sorting\combined_features.csv")

# Create Age Group Dataset (Remove 'filename' and 'gender')
age_group_df = df.drop(columns=['filename', 'gender'])
age_group_df.to_csv("age_group_features.csv", index=False)

# Create Gender Dataset (Remove 'filename' and 'age_group')
gender_df = df.drop(columns=['filename', 'age_group'])
gender_df.to_csv("gender_features.csv", index=False)
