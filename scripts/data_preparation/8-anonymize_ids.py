# Task:   Prepare the data release: replace author IDs with sequential IDs that keep the order of
#         the original IDs, and event IDs with random sequential IDs (one mapping for all files),
#         and check that no identifying columns remain.
# Input:  data/authors_filtered.csv and the release files listed in `files` below
# Output: <OUT_DIR>/ with the same file names
# Run:    python 8-anonymize_ids.py ../../data <OUT_DIR>

import gzip
import polars as pl
import numpy as np
import sys
import os
from os.path import join

src = sys.argv[1]
out = sys.argv[2]
if os.path.realpath(src) == os.path.realpath(out):
    sys.exit("ERROR: <OUT_DIR> must differ from <DATA_DIR> (the input files would be overwritten).")
os.makedirs(out, exist_ok=True)

# (filename, ID columns to map, gzip output)
files = [
    ("authors_filtered.csv",                            ["author_id"], False),
    ("Austria-Panel-Daily-de_en.csv",                   [], False),
    ("Austria-Panel-Daily-de.csv",                      [], False),
    ("Austria-Panel-Daily-all.csv",                     [], False),
    ("Austria-Panel-Daily-Thresholds.csv",              [], False),
    ("Austria-Panel-Users-Features-Tweet-de_en.csv",    ["author_id"], False),
    ("Austria-Panel-Users-Features-Tweet-de.csv",       ["author_id"], False),
    ("Austria-Panel-Users-Daily.csv.gz",                ["author_id"], True),
    ("Austria-Panel-User-Statistics.csv",               ["author_id"], False),
    ("Austria-Panel-Supersharers.csv",                  ["author_id"], False),
    ("Austria-Panel-Supersharers-Groups.csv",           ["author_id"], False),
    ("Austria-Panel-Windows-1h-10min-de_en.csv.gz",     ["author_id", "event_id"], True),
    ("Austria-Panel-Windows-24h-10min-de_en.csv.gz",    ["author_id", "event_id"], True),
]
# columns that must not appear in a released file unless they are mapped
ID_COLS = {"id", "author_id", "event_id", "conversation_id", "in_reply_to_user_id",
           "username", "name", "description", "location", "expanded_urls", "text",
           "text_cleaned", "ng_domain", "url", "domain"}

rng = np.random.default_rng(seed=None)  # no fixed seed: the mapping is not reproducible by design

# author_id mapping from the filtered user list
authors = (pl.read_csv(join(src, "authors_filtered.csv"),
                       columns=["author_id"], schema_overrides={"author_id": pl.Utf8})
             ["author_id"].unique().to_list())
# Order-preserving: anon ID i goes to the i-th real ID in text order (C locale, which is how
# dplyr::group_by() sorts character IDs). Scripts that break ties by numeric author_id (e.g. the
# supersharer cutoff in 11-analyze_supersharers.Rmd) therefore resolve ties exactly as in the
# analysis with the original IDs. The order only mirrors the order of the original IDs (roughly
# account age), which User-Statistics.csv already contains as the account creation date.
id_map = {real: i + 1 for i, real in enumerate(sorted(authors))}  # 1-indexed integers
print(f"Built mapping for {len(id_map):,} users.")

# event_id mapping across all window files (same news tweet -> same anon ID in 1h and 24h)
event_ids = set()
for filename, id_cols, _ in files:
    path = join(src, filename)
    if "event_id" in id_cols and os.path.exists(path):
        event_ids |= set(pl.read_csv(path, columns=["event_id"],
                                     schema_overrides={"event_id": pl.Utf8})
                           ["event_id"].unique().to_list())
event_ids = sorted(event_ids)
event_id_map = {real: int(anon) for real, anon in
                zip(event_ids, rng.permutation(len(event_ids)) + 1)}
print(f"Built mapping for {len(event_id_map):,} event IDs.")


def apply_mapping(df: pl.DataFrame, col: str, mapping: dict, filename: str) -> pl.DataFrame:
    vals = df[col].cast(pl.Utf8)
    unmapped = vals.drop_nulls().filter(~vals.drop_nulls().is_in(list(mapping)))
    if len(unmapped) > 0:
        sys.exit(f"ERROR: {len(unmapped):,} {col} values in {filename} are not in the "
                 f"mapping. Nothing written for this file; check the input.")
    return df.with_columns(
        vals.replace_strict(mapping, return_dtype=pl.Int64).alias(col))


for filename, id_cols, compressed in files:
    path = join(src, filename)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {filename}")
        continue

    # read IDs as strings (only columns present; polars would add missing ones)
    header = pl.read_csv(path, n_rows=0).columns
    id_overrides = {c: pl.Utf8 for c in ["author_id", "event_id"] if c in header}
    df = pl.read_csv(path, infer_schema_length=10000, schema_overrides=id_overrides)

    if filename == "authors_filtered.csv":
        # drop username, name, description, location and expanded_urls
        # to prevent re-identification via these fields (as in v1)
        df = df.drop(["username", "name", "description", "location", "expanded_urls"],
                     strict=False)

    if filename == "Austria-Panel-Users-Daily.csv.gz":
        # the emotion columns are not used by any released script; drop them so the
        # release does not contain emotion counts under a different definition
        df = df.drop(["n_anger", "n_fear", "prop_anger", "prop_fear"], strict=False)

    for col in id_cols:
        if col in df.columns:
            mapping = id_map if col == "author_id" else event_id_map
            df = apply_mapping(df, col, mapping, filename)

    leftover = (ID_COLS & set(df.columns)) - set(id_cols)
    if leftover:
        sys.exit(f"ERROR: {filename} contains identifying columns {sorted(leftover)}")

    out_path = join(out, filename)
    if compressed:
        with gzip.open(out_path, "wb") as f:
            f.write(df.write_csv().encode())
    else:
        df.write_csv(out_path)

    print(f"  Saved: {filename} ({df.height:,} rows)")

print(f"\nAnonymized files written to {out}/")
