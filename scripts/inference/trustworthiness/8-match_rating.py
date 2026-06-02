# ===== SCRIPT OVERVIEW =====
# Task:   Match NewsGuard domain ratings to tweets by domain and month to produce
#         the central analysis-ready file used by all downstream analysis scripts.
#         Ratings from 2021+ matched on (domain, month); earlier tweets on domain only.
#         Duplicate domain rows aggregated: Score averaged, Orientation joined.
# Input:  <DATA_DIR>/Austria-Panel-Tweets.csv.gz
#         <DATA_DIR>/urls/matched_domains_all.csv.gz   (output of 7-match_domains.py)
#         <DATA_DIR>/dtypes.pkl
#         <DATA_DIR>/newsguard/domains_unique.csv                  (proprietary — do not release)
# Output: <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         Columns added: ng_domain, Score (0–100), Orientation
# Usage:  python 8-match_rating.py <DATA_DIR>

import pandas as pd
from os.path import join
import pickle as pkl
from tqdm import tqdm
import sys

src = sys.argv[1] # /misinfo-panel-austria/data/
file = "Austria-Panel-Tweets.csv.gz"
chunk_size = 10_000_000
n_chunks = None  # unknown without decompressing; tqdm shows count only

with open(join(src, "dtypes.pkl"), "rb") as f:
    dtype = pkl.load(f)

# load urls
domains = pd.read_csv(join(src,"urls",
                        "matched_domains_all.csv.gz"),
                        low_memory=False,
                        compression="gzip")

ratings = pd.read_csv(join(src,"newsguard",
                        "domains_unique.csv"),
                        low_memory=False)

ratings["file_month"] = pd.to_datetime(
        ratings["file_month"]).dt.tz_localize(None).dt.to_period("M")

# keep ratings from after 2021
stable_ratings = ratings[ratings["file_month"] >= "2021-01"]
first_ratings = stable_ratings["file_month"].min()

first_header = True
for chunk in tqdm(pd.read_csv(join(src, file),
                              dtype=dtype,
                              compression="gzip",
                              low_memory=False,
                              chunksize=chunk_size),
                  total=n_chunks,
                  desc="Processing chunks"):
    
    # explode urls
    chunk_exploded = chunk.copy()
    chunk_exploded["expanded_urls"] = chunk_exploded["expanded_urls"].astype(str)
    chunk_exploded["url"] = chunk_exploded["expanded_urls"].str.split(" ")
    chunk_exploded = chunk_exploded.explode("url")
    
    # merge domains back with URLs 
    chunk_merged = chunk_exploded\
        .merge(domains, 
               on="url", 
               how="left")

    # merge with newsguard based on domain and date
    chunk_merged["month"] = pd.to_datetime(
        chunk_merged["created_at"]).dt.tz_localize(None).dt.to_period("M")
    
    # filter for ratings after 2021
    chunk_after = chunk_merged[
        chunk_merged["month"] >= first_ratings
    ].copy()

    chunk_before = chunk_merged[
        chunk_merged["month"] < first_ratings
    ].copy()


    chunk_rated_after = chunk_after.merge(
        stable_ratings,
        left_on=["ng_domain", "month"],
        right_on=["Domain", "file_month"],
        how="left"
    )

    chunk_rated_before = chunk_before.merge(
        stable_ratings,
        left_on=["ng_domain"],
        right_on=["Domain"],
        how="left"
    )

    chunk_rated = pd.concat([
        chunk_rated_after,
        chunk_rated_before
    ], ignore_index=True)

    # merge duplicated rows
    rated_score = chunk_rated[chunk_rated["Domain"].notna()] 

    dups_rated = rated_score[rated_score\
                .duplicated([
                    "id", "author_id"],
                keep=False)]

    agg_dups = (dups_rated\
                 .groupby("id")
                 .agg({
                    "urls": lambda x: " ".join(x), # join urls
                    "ng_domain": lambda x: ", ".join(x), # join domains
                    "Orientation": lambda x: ", ".join(x), # join if multiple
                    "Score": "mean" # average score
                    })
                .reset_index()
    )

    # merge back with full rows
    dups_rated = dups_rated.drop(columns=["ng_domain", "Score", "Orientation"])
    dups_rated = dups_rated.merge(
        agg_dups[["id", "ng_domain", "Score", "Orientation"]],
        on="id",
        how="left"
    )
    # keep unique rows with multiple domains
    dup_unique = (dups_rated\
        .drop(columns=["ng_domain", "Score", "Orientation"])
        .merge(
            agg_dups[["id", "ng_domain", "Score", 'Orientation']],
            on="id",
            how="left")
        .drop_duplicates(subset=["id", "author_id"])
    )
    
    # keep the ones not in duplicated
    chunk_unique = chunk_rated[~chunk_rated["id"].isin(dup_unique["id"])]

    # concatenate unique and rated rows
    chunk_final = pd.concat([
        chunk_unique,
        dups_rated
    ], ignore_index=True)

    # drop duplicates
    chunk_final = chunk_final\
        .drop_duplicates(subset=["id", "author_id"])

    # save to csv
    chunk_final.to_csv(join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"), 
                mode="a", 
                header=first_header,
                index=False, compression="gzip")
    
    first_header = False
    

print(f"Saved data to {join(src, 'Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz')}")