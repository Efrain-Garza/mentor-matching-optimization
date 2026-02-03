from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MENTEE_XLSX = Path("data/raw/Mentee Data.xlsx")
DOMAIN_VEC_PATH = Path("data/features/utsa5_domain_vectors.parquet")
OUT_CSV = Path("outputs/utsa5_first5_scores.csv")

# IMPORTANT: Your sheet has a trailing space in this column name.
TEXT_COLS = [
    "Career Goals",
    "Program Goals",
    "Fields of Interest",
    "Additional Info to Consider ",
]

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_narrative(row: pd.Series) -> str:
    parts = []
    for c in TEXT_COLS:
        v = row.get(c, "")
        if pd.notna(v):
            parts.append(str(v))
    return normalize_text(" ".join(parts))

def main() -> None:
    # 1) Load domain vectors
    if not DOMAIN_VEC_PATH.exists():
        raise FileNotFoundError(f"Missing domain vectors: {DOMAIN_VEC_PATH.resolve()}")

    dv = pd.read_parquet(DOMAIN_VEC_PATH)

    # Build domain "documents" from top terms
    top_k = 80
    domain_docs = (
        dv.sort_values(["domain", "weight"], ascending=[True, False])
          .groupby("domain", as_index=False)
          .head(top_k)
          .groupby("domain")["term"]
          .apply(lambda x: " ".join(x.tolist()))
          .reset_index()
          .rename(columns={"term": "doc"})
    )
    domain_docs["doc"] = domain_docs["doc"].apply(normalize_text)

    # 2) Load mentees
    if not MENTEE_XLSX.exists():
        raise FileNotFoundError(f"Missing mentee file: {MENTEE_XLSX.resolve()}")

    df = pd.read_excel(MENTEE_XLSX)
    df5 = df.head(5).copy()
    df5["narrative"] = df5.apply(build_narrative, axis=1)

    # 3) Joint TF-IDF space (domains + mentees)
    corpus = domain_docs["doc"].tolist() + df5["narrative"].tolist()

    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        max_df=0.90,
    )
    X = vec.fit_transform(corpus)

    X_dom = X[: len(domain_docs)]
    X_men = X[len(domain_docs):]

    # 4) Similarity
    sims = cosine_similarity(X_men, X_dom)  # (5, num_domains)

    # 5) Output
    rows = []
    for i in range(len(df5)):
        scores = sims[i]
        order = np.argsort(-scores)

        top1_i = int(order[0])
        top2_i = int(order[1]) if len(order) > 1 else int(order[0])

        mentee_name = str(df5.iloc[i].get("First Name", f"Row{i+1}")).strip()
        major = str(df5.iloc[i].get("Major", "")).strip()

        rows.append({
            "mentee": mentee_name,
            "major": major,
            "top_domain": domain_docs.loc[top1_i, "domain"],
            "top_score": float(scores[top1_i]),
            "runner_up": domain_docs.loc[top2_i, "domain"],
            "runner_up_score": float(scores[top2_i]),
        })

    out = pd.DataFrame(rows).sort_values(["top_score"], ascending=False)

    print("\nScoring first 5 mentees vs UTSA5 domains:")
    print(out.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
