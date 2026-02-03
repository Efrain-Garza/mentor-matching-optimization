from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

IN_PATH = Path("data/features/ipod_fun.parquet")
OUT_PATH = Path("data/features/utsa5_domain_vectors.parquet")

# These are retrieval anchors, not the domain definition.
DOMAIN_ANCHORS: Dict[str, set[str]] = {
    "Accounting": {
        "account", "accounting", "audit", "tax", "controller", "controls",
        "ledger", "payable", "receivable", "payroll", "compliance"
    },
    "Economics": {
        "economics", "economic", "policy", "macro", "micro", "trade",
        "research", "market", "markets", "analysis"
    },
    "Finance": {
        "finance", "financial", "investment", "treasury", "risk",
        "controller", "markets", "trade", "capital", "portfolio"
    },
    "Management": {
        "management", "manager", "operations", "strategy", "strategic",
        "leadership", "project", "program", "business", "planning"
    },
    "Marketing": {
        "marketing", "brand", "digital", "advertising", "campaign",
        "customer", "content", "social", "communications", "sales"
    },
}

# Remove generic words that appear across many domains (organizational boilerplate).
GENERIC_STOP = {
    "team", "department", "division", "office", "company", "corporate",
    "administration", "support", "services", "service", "solutions",
    "system", "systems", "technology", "technical", "group", "unit",
    "manager", "management", "director", "senior", "associate", "lead",
    "assistant", "specialist", "analyst", "coordinator", "executive",
}

def require_anchor_hits(fun_text: str, anchors: set[str], min_hits: int) -> bool:
    toks = fun_text.split()
    hits = 0
    for a in anchors:
        if a in toks:
            hits += 1
            if hits >= min_hits:
                return True
    return False

def build_domain_vector(
    fun_text_series: pd.Series,
    anchors: set[str],
    *,
    min_anchor_hits: int = 2,
    min_df: int = 10,
    max_df: float = 0.50,
) -> Tuple[pd.DataFrame, int]:
    # Seeded retrieval: subset of IPOD FUN texts relevant to this domain
    mask = fun_text_series.fillna("").apply(lambda s: require_anchor_hits(s, anchors, min_anchor_hits))
    sub = fun_text_series.loc[mask].fillna("")

    # TF-IDF fit only on the subset; centroid becomes the domain vector.
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        max_df=max_df,
    )
    X = vec.fit_transform(sub.values)
    centroid = np.asarray(X.mean(axis=0)).ravel()
    vocab = vec.get_feature_names_out()

    df_vec = pd.DataFrame({"term": vocab, "weight": centroid})
    df_vec = df_vec.sort_values("weight", ascending=False)

    # Clean: remove generic org boilerplate (keeps vector more diagnostic)
    df_vec_clean = df_vec[~df_vec["term"].isin(GENERIC_STOP)].reset_index(drop=True)

    return df_vec_clean, int(mask.sum())

def main() -> None:
    df = pd.read_parquet(IN_PATH)
    fun_text = df["fun_text"].fillna("")

    all_rows: List[pd.DataFrame] = []

    print("Loaded IPOD FUN artifact rows:", len(df))
    print("Input:", IN_PATH.resolve())
    print()

    for domain, anchors in DOMAIN_ANCHORS.items():
        vec_df, n_sub = build_domain_vector(fun_text, anchors, min_anchor_hits=2)

        print(f"=== {domain} ===")
        print("Seeded subset rows:", n_sub)
        print("Top terms:")
        for _, r in vec_df.head(20).iterrows():
            print(f"  {r['term']:<18s} {r['weight']:.6f}")
        print()

        out = vec_df.copy()
        out.insert(0, "domain", domain)
        out.insert(1, "subset_rows", n_sub)
        all_rows.append(out)

    out_all = pd.concat(all_rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_all.to_parquet(OUT_PATH, index=False)

    print("Wrote:", OUT_PATH.resolve())
    print("Rows in domain-vector table:", len(out_all))
    print("Columns:", list(out_all.columns))

if __name__ == "__main__":
    main()
