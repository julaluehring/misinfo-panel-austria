# ===== SCRIPT OVERVIEW =====
# Task:   Build the domain co-sharing network among misinformation supersharers.
#         Creates nodes (domains) and edges (co-shared by same user).
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/Austria-Panel-Supersharers.csv  (output of 11-analyze_supersharers.Rmd)
# Output: <DATA_DIR>/Austria-Panel-Source-Nodes.csv
#         <DATA_DIR>/Austria-Panel-Source-Edges.csv
# Usage:  python 12c-pull_sources.py <DATA_DIR>

import polars as pl
import sys
from os.path import join
from itertools import combinations
import pandas as pd

src = sys.argv[1]

# load the supersharer authors
ids = pd.read_csv(join(src, "Austria-Panel-Supersharers.csv"),
                  usecols=["author_id"])
authors = set(ids["author_id"].astype(int).tolist())

# Load tweet data from observation period
tweets_df = (
    pl.scan_csv(
        join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"),
        low_memory=True
    )
    .select(["author_id", "created_at", "domain", "Score"])
    .with_columns([pl.col("created_at").str.to_datetime(time_zone="UTC"),
                  (pl.col("Score") >= 60).cast(pl.Int8).alias("Rating")])
    .filter(pl.col("created_at") >= pl.datetime(2019, 1, 1, time_zone="UTC"))
    .filter(pl.col("created_at") < pl.datetime(2023, 4, 1, time_zone="UTC"))
).collect()

# filter to tweets by supersharer authors with valid domains
tweets_df = tweets_df.filter(
    pl.col("author_id").is_in(authors) &
    pl.col("domain").is_not_null() &
    pl.col("Score").is_not_null()
)

# create nodes
nodes_df = (
    tweets_df
    .group_by("domain")
    .agg([
        pl.len().alias("mentions"),
        pl.col("Rating").mode().first().alias("Rating"),
        pl.n_unique("author_id").alias("unique_users")
    ])
)

nodes_df.write_csv(join(src, "Austria-Panel-Source-Nodes.csv"))
print(f"Saved nodes to {join(src, 'Austria-Panel-Source-Nodes.csv')}")

# per domain, aggregate the number of mentions by users
user_domains = (
    tweets_df
    .group_by("author_id")
    .agg(pl.col("domain").unique().alias("domains"))
)

# generate co-exposure edges
edge_list = []
for row in user_domains.iter_rows(named=True):
    domains = row["domains"]
    if len(domains) >= 2:
        for source1, source2 in combinations(sorted(domains), 2):
            edge_list.append({"source": source1, "target": source2})

source_edges = (
    pl.DataFrame(edge_list)
    .group_by(["source", "target"])
    .agg(pl.len().alias("weight"))
    .sort("weight", descending=True)
)

source_edges.write_csv(join(src, "Austria-Panel-Source-Edges.csv"))
print(f"Saved edges to {join(src, 'Austria-Panel-Source-Edges.csv')}")
