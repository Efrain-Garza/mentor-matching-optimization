from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

IN_PATH = Path("data/features/ipod_fun.parquet")

# Anchors are not the domain definition — they are just a retrieval filter.
FINANCE_ANCHORS = {
    "finance", "financial", "account", "accounting", "audit",
    "investment", "treasury", "risk", "controller"
}

def main() -> None:
    df = pd.read_parquet(IN_PATH)
    fun_text = df["fun_text"].fillna("")

    # Build a finance-relevant subset (seeded retrieval)
    mask = fun_text.apply(lambda s: any(a in s.split() for a in FINANCE_ANCHORS))
    sub = fun_text.loc[mask]

    print("IPOD rows:", len(df))
    print("Finance-seeded subset rows:", len(sub))

    # TF-IDF on the subset; centroid is the "domain vector"
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=10,
        max_df=0.50
    )
    X = vec.fit_transform(sub.values)
    centroid = np.asarray(X.mean(axis=0)).ravel()
    vocab = vec.get_feature_names_out()

    top_idx = np.argsort(-centroid)[:50]
    print("\nTop Finance-associated FUN terms (TF-IDF centroid):")
    for i in top_idx:
        print(f"{vocab[i]:20s} {centroid[i]:.6f}")

if __name__ == "__main__":
    main()

