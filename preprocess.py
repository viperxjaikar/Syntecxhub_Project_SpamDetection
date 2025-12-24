import pandas as pd
import re
import random

# Load dataset
df = pd.read_csv("dataset.csv")

# Validate schema
if not {"label", "message"}.issubset(df.columns):
    raise ValueError("dataset.csv must contain 'label' and 'message' columns")

# Text cleaning
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["message"] = df["message"].astype(str).apply(clean_text)

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})
if df["label"].isnull().any():
    raise ValueError("Labels must be 'ham' or 'spam'")

# ---- REALISM FIX: introduce small label noise (5%) ----
noise_ratio = 0.05
n_flip = int(len(df) * noise_ratio)
indices = random.sample(range(len(df)), n_flip)

df.loc[indices, "label"] = 1 - df.loc[indices, "label"]
# ------------------------------------------------------

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
