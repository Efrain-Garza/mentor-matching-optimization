# scripts/ipod_fun_extraction_cudf_fast.py

import cudf
from pathlib import Path
import time

IPOD_PATH = Path.home() / "workspace/datasets/IPOD/data/ipod_ner.csv"

t0 = time.perf_counter()

# ---------------------------
# Load data
# ---------------------------
t_load = time.perf_counter()
gdf = cudf.read_csv(IPOD_PATH)
t_load_end = time.perf_counter()

# ---------------------------
# Tokenize
# ---------------------------
gdf["tokens"] = gdf["Processed_Title"].str.split()
gdf["tags"] = gdf["Tag_A1"].str.split()

# ---------------------------
# Explode (cuDF-safe)
# ---------------------------
t_explode = time.perf_counter()
gdf_tok = gdf.explode("tokens")
gdf_tok = gdf_tok.explode("tags")
t_explode_end = time.perf_counter()

# ---------------------------
# Filter FUN tokens
# ---------------------------
t_filter = time.perf_counter()
gdf_fun = gdf_tok[gdf_tok["tags"] == "FUN"]
t_filter_end = time.perf_counter()

# ---------------------------
# Aggregate FUN tokens back to title-level
# ---------------------------
t_group = time.perf_counter()

fun_by_title = (
    gdf_fun
    .groupby(level=0)["tokens"]
    .agg(list)
)

gdf["fun_tokens"] = fun_by_title

# cuDF-safe null handling
mask = gdf["fun_tokens"].isnull()
gdf.loc[mask, "fun_tokens"] = cudf.Series([[]] * mask.sum())

t_group_end = time.perf_counter()

# ---------------------------
# Output
# ---------------------------
print(gdf[["Processed_Title", "Tag_A1", "fun_tokens"]].head(5))
print("\nTotal rows:", len(gdf))

print("\nTiming (seconds):")
print(f"  Load CSV       : {t_load_end - t_load:.3f}")
print(f"  Explode tokens : {t_explode_end - t_explode:.3f}")
print(f"  Filter FUN     : {t_filter_end - t_filter:.3f}")
print(f"  Group collect  : {t_group_end - t_group:.3f}")
print(f"  Total runtime  : {t1 - t0:.3f}")
