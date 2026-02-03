# scripts/ipod_fun_extraction.py

import pandas as pd
from pathlib import Path

# ---------------------------
# Paths
# ---------------------------
IPOD_PATH = Path.home() / "workspace/datasets/IPOD/data/ipod_ner.csv"

# ---------------------------
# Load IPOD
# ---------------------------
df = pd.read_csv(IPOD_PATH)

# ---------------------------
# FUN token extraction
# ---------------------------
def extract_fun_tokens(processed_title: str, tag_str: str):
    tokens = processed_title.split()
    tags = tag_str.split()
    return [t for t, g in zip(tokens, tags) if g == "FUN"]

df["fun_tokens"] = df.apply(
    lambda r: extract_fun_tokens(r["Processed_Title"], r["Tag_A1"]),
    axis=1
)

# ---------------------------
# Sanity check
# ---------------------------
print(df[["Processed_Title", "Tag_A1", "fun_tokens"]].head(5))
print("\nTotal rows:", len(df))
