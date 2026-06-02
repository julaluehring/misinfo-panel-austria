# ===== SCRIPT OVERVIEW =====
# Task:   Replace real author_id and tweet event_id values with randomized sequential
#         integers consistently across all releasable data files.
#         Output files are written to <DATA_DIR>/release/ and are safe to publish.
# Input:  <DATA_DIR>/authors_filtered.csv
#         <DATA_DIR>/Austria-Panel-Users-Features-Tweet.csv
#         <DATA_DIR>/Austria-Panel-Users-Daily.csv.gz
#         <DATA_DIR>/Austria-Panel-User-Statistics.csv
#         <DATA_DIR>/Austria-Panel-Supersharers.csv
#         <DATA_DIR>/Austria-Panel-Tweets-Windows-1h-10min.csv.gz
#         <DATA_DIR>/Austria-Panel-Tweets-Windows-24h-10min.csv.gz
# Output: <DATA_DIR>/release/<same filenames>  (author_id replaced with anon_id)
# Usage:  python 8-anonymize_ids.py <DATA_DIR> <DATA_DIR>/release

import gzip
import polars as pl
import numpy as np
import sys
import os
from os.path import join

src = sys.argv[1]
out = sys.argv[2]

# build mapping from authors_filtered.csv (the filtered user list)
authors = pl.read_csv(join(src, "authors_filtered.csv")).select(
    pl.col("author_id").cast(pl.Utf8)
)["author_id"].to_list()

rng = np.random.default_rng(seed=None)  # no fixed seed — mapping is not reproducible by design
shuffled = rng.permutation(len(authors)) + 1  # 1-indexed integers
id_map = {real: int(anon) for real, anon in zip(authors, shuffled)}

print(f"Built mapping for {len(id_map):,} users.")

# build event_id mapping — read only the event_id column to keep pre-scan lightweight
event_ids = []
for fname in ["Austria-Panel-Windows-1h-10min.csv.gz",
              "Austria-Panel-Windows-24h-10min.csv.gz"]:
    path = join(src, fname)
    if os.path.exists(path):
        ids = (pl.read_csv(path, columns=["event_id"], infer_schema_length=10000)
                 ["event_id"].cast(pl.Utf8).unique().to_list())
        event_ids.extend(ids)

event_ids = list(set(event_ids))
shuffled_events = rng.permutation(len(event_ids)) + 1
event_id_map = {real: int(anon) for real, anon in zip(event_ids, shuffled_events)}
print(f"Built mapping for {len(event_id_map):,} event IDs.")


def apply_mapping(df: pl.DataFrame, col: str, mapping: dict) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).cast(pl.Utf8).replace(mapping).cast(pl.Int64).alias(col)
    )


files = [
    ("authors_filtered.csv",                    ["author_id"], False),
    ("Austria-Panel-Daily.csv",                 ["author_id"], False),
    ("Austria-Panel-Users-Features-Tweet.csv",  ["author_id"], False),
    ("Austria-Panel-Users-Daily.csv.gz",        ["author_id"], True),
    ("Austria-Panel-User-Statistics.csv",       ["author_id"], False),
    ("Austria-Panel-Supersharers.csv",          ["author_id"], False),
    ("Austria-Panel-Windows-1h-10min.csv.gz",   ["author_id", "event_id"], True),
    ("Austria-Panel-Windows-24h-10min.csv.gz",  ["author_id", "event_id"], True),
]

for filename, id_cols, compressed in files:
    path = join(src, filename)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {filename}")
        continue

    df = pl.read_csv(path, infer_schema_length=10000)
    for col in id_cols:
        if col in df.columns:
            mapping = id_map if col == "author_id" else event_id_map
            df = apply_mapping(df, col, mapping)

    out_path = join(out, filename)
    if filename == "authors_filtered.csv":
        # remove cols username, name, description, location, expanded_urls
        # to prevent any potential re-identification via these fields
        df = df.drop(["username", "name", "description", "location", "expanded_urls"])
    if compressed:
        with gzip.open(out_path, "wb") as f:
            f.write(df.write_csv().encode())
    else:
        df.write_csv(out_path)

    print(f"  Saved: {filename}")

print(f"\nAnonymized files written to {out}/")
