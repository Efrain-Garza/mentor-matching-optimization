"""
Build IPOD FUN-token artifact (hybrid GPU+CPU).

- GPU (cuDF): fast CSV ingest
- CPU (pandas): correct, simple FUN extraction (token/tag alignment)
- Output: Parquet with columns:
    - processed_title (str)
    - tag_a1 (str)
    - fun_tokens (list[str])
    - fun_text (str) space-joined fun tokens
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cudf
import pandas as pd

IPOD_PATH = Path.home() / "workspace/datasets/IPOD/data/ipod_ner.csv"

# Use your existing folder structure: data/features
OUT_PATH = Path("data/features/ipod_fun.parquet")


def extract_fun_tokens(processed_title: str, tag_a1: str) -> List[str]:
    if not isinstance(processed_title, str) or not isinstance(tag_a1, str):
        return []
    tokens = processed_title.split()
    tags = tag_a1.split()
    n = min(len(tokens), len(tags))
    return [tok for tok, tg in zip(tokens[:n], tags[:n]) if tg == "FUN"]


def main() -> None:
    t0 = time.perf_counter()

    # 1) GPU ingest (fast)
    t_load0 = time.perf_counter()
    gdf = cudf.read_csv(IPOD_PATH, usecols=["Processed_Title", "Tag_A1"])
    t_load1 = time.perf_counter()

    # 2) Move needed columns to CPU
    t_to_cpu0 = time.perf_counter()
    pdf = gdf.to_pandas()
    t_to_cpu1 = time.perf_counter()

    # 3) CPU FUN extraction (correct)
    t_ext0 = time.perf_counter()
    pt = pdf["Processed_Title"].astype(str).tolist()
    tg = pdf["Tag_A1"].astype(str).tolist()
    fun_tokens = [extract_fun_tokens(a, b) for a, b in zip(pt, tg)]

    pdf = pd.DataFrame(
        {
            "processed_title": pt,
            "tag_a1": tg,
            "fun_tokens": fun_tokens,
            "fun_text": [" ".join(x) for x in fun_tokens],
        }
    )
    t_ext1 = time.perf_counter()

    # 4) Persist artifact
    t_out0 = time.perf_counter()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(OUT_PATH, index=False)
    t_out1 = time.perf_counter()

    t1 = time.perf_counter()

    print(pdf[["processed_title", "tag_a1", "fun_tokens"]].head(5))
    print("\nRows:", len(pdf))
    print("\nTiming (seconds):")
    print(f"  GPU read_csv        : {t_load1 - t_load0:.3f}")
    print(f"  to_pandas           : {t_to_cpu1 - t_to_cpu0:.3f}")
    print(f"  FUN extraction (CPU): {t_ext1 - t_ext0:.3f}")
    print(f"  write_parquet       : {t_out1 - t_out0:.3f}")
    print(f"  total               : {t1 - t0:.3f}")
    print(f"\nWrote: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
