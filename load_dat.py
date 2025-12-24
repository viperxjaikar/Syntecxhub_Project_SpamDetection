import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Basic validation
required_columns = {"label", "message"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain columns: {required_columns}")

# Inspect dataset
print(df.shape)
print(df["label"].value_counts())
print(df.isnull().sum())

# Preview
print(df.head())
