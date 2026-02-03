from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MENTOR_PATH = Path("data/cleaned/mentor_clean.parquet")
DOMAIN_VEC_PATH = Path("data/features/utsa5_domain_vectors.parquet")
OUT_PATH = Path("data/features/mentor_domain_profiles.parquet")

DOMAINS = ["Accounting", "Economics", "Finance", "Management", "Marketing"]

def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main() -> None:
    mentors = pd.read_parquet(MENTOR_PATH)
    dv = pd.read_parquet(DOMAIN_VEC_PATH)

    # --- Build domain documents ---
    domain_docs = (
        dv.sort_values(["domain", "weight"], ascending=[True, False])
          .groupby("domain")
          .head(80)
          .groupby("domain")["term"]
          .apply(lambda x: " ".join(x.tolist()))
          .reset_index()
    )
    domain_docs["doc"] = domain_docs["term"].apply(normalize)

    # --- Mentor text channels ---
    mentors["structural_text"] = (
        mentors["standardized_degree"].fillna("") + " " +
        mentors["Job Title"].fillna("") + " " +
        mentors["Current Role"].fillna("")
    ).apply(normalize)

    mentors["interest_text"] = (
        mentors["Field of Interest"].fillna("") + " " +
        mentors["Program Goals"].fillna("")
    ).apply(normalize)

    # --- TF-IDF joint space ---
    corpus = (
        domain_docs["doc"].tolist() +
        mentors["structural_text"].tolist() +
        mentors["interest_text"].tolist()
    )

    vec = TfidfVectorizer(min_df=2, max_df=0.85)
    X = vec.fit_transform(corpus)

    n_dom = len(domain_docs)
    n_men = len(mentors)

    X_dom = X[:n_dom]
    X_struct = X[n_dom : n_dom + n_men]
    X_interest = X[n_dom + n_men :]

    # --- Similarities ---
    sim_struct = cosine_similarity(X_struct, X_dom)
    sim_interest = cosine_similarity(X_interest, X_dom)

    results = []
    for i, row in mentors.iterrows():
        p_idx = int(np.argmax(sim_struct[i]))
        s_idx = int(np.argmax(sim_interest[i]))

        primary_domain = domain_docs.loc[p_idx, "domain"]
        secondary_domain = domain_docs.loc[s_idx, "domain"]

        results.append({
            "First Name": row["First Name"],
            "standardized_degree": row["standardized_degree"],
            "primary_domain": primary_domain,
            "primary_score": float(sim_struct[i, p_idx]),
            "secondary_domain": secondary_domain,
            "secondary_score": float(sim_interest[i, s_idx]),
            "human_vs_model_agree": normalize(row["standardized_degree"]) == normalize(primary_domain)
        })

    out = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(out.head(10).to_string(index=False))
    print("\nAgreement rate:",
          out["human_vs_model_agree"].mean().round(3))

    print("\nWrote:", OUT_PATH.resolve())

if __name__ == "__main__":
    main()
