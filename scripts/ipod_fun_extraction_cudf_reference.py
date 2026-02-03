# scripts/ipod_fun_extraction_cudf.py

import cudf
from pathlib import Path
import time

# ---------------------------
# Paths
# ---------------------------
IPOD_PATH = Path.home() / "workspace/datasets/IPOD/data/ipod_ner.csv"

t0 = time.perf_counter()

# ---------------------------
# Load IPOD on GPU
# ---------------------------
t_load_start = time.perf_counter()
gdf = cudf.read_csv(IPOD_PATH)
t_load_end = time.perf_counter()

# ---------------------------
# FUN token extraction
# ---------------------------
def extract_fun_tokens_gpu(processed, tags):
    tokens = processed.split()
    tag_list = tags.split()
    return [t for t, g in zip(tokens, tag_list) if g == "FUN"]

t_extract_start = time.perf_counter()

gdf["fun_tokens"] = gdf.apply_rows(
    lambda processed, tags: extract_fun_tokens_gpu(processed, tags),
    incols=["Processed_Title", "Tag_A1"],
    outcols=dict(fun_tokens=list),
)

t_extract_end = time.perf_counter()
t1 = time.perf_counter()

# ---------------------------
# Sanity check
# ---------------------------
print(gdf[["Processed_Title", "Tag_A1", "fun_tokens"]].head(5))
print("\nTotal rows:", len(gdf))

# ---------------------------
# Timing summary
# ---------------------------
print("\nTiming (seconds):")
print(f"  Load CSV      : {t_load_end - t_load_start:.3f}")
print(f"  FUN extraction: {t_extract_end - t_extract_start:.3f}")
print(f"  Total runtime : {t1 - t0:.3f}")
