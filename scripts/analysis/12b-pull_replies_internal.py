# ===== SCRIPT OVERVIEW =====
# Task:   Build the internal reply network (edges only between users within the sample).
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/Austria-Panel-Users-Daily.csv.gz
# Output: <DATA_DIR>/Austria-Panel-Network-Edges-Internal.csv
#         <DATA_DIR>/Austria-Panel-Network-Nodes-Internal.csv
# Usage:  python 12b-pull_replies_internal.py <DATA_DIR>

import polars as pl
import sys
from os.path import join

src = sys.argv[1]

# Load the sample authors
authors = pl.read_csv(
    join(src, "Austria-Panel-Users-Daily.csv.gz"),
    columns=["author_id"]
)["author_id"].unique().to_list()

authors_set = set(authors)
print(f"Total authors in sample: {len(authors)}")

# Load reply data from observation period
replied_to_df = (
    pl.scan_csv(
        join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"),
        low_memory=True
    )
    .select(["in_reply_to_user_id", "created_at", "author_id"])
    .filter(pl.col("in_reply_to_user_id").is_not_null())
    .with_columns(pl.col("created_at").str.to_datetime(time_zone="UTC"))
    .filter(pl.col("created_at") >= pl.datetime(2019, 1, 1, time_zone="UTC"))
    .filter(pl.col("created_at") < pl.datetime(2023, 4, 1, time_zone="UTC"))
).collect()

print(f"Total replies in period: {len(replied_to_df)}")

# Filter to ONLY internal replies (both source and target in sample)
internal_replies = replied_to_df.filter(
    pl.col("author_id").is_in(authors) &
    pl.col("in_reply_to_user_id").is_in(authors)
)

print(f"Internal replies (sample -> sample): {len(internal_replies)}")

# Create edges: aggregate reply counts between user pairs
internal_edges = (
    internal_replies
    .group_by(["author_id", "in_reply_to_user_id"])
    .agg(pl.len().alias("weight"))
    .sort("weight", descending=True)
)

print(f"Unique internal edges: {len(internal_edges)}")

# Create nodes: all users involved in internal network
internal_user_ids = set(internal_edges["author_id"].unique().to_list()) | \
                    set(internal_edges["in_reply_to_user_id"].unique().to_list())

print(f"Users in internal network: {len(internal_user_ids)}")

# Calculate node metrics
replies_received = (
    internal_replies
    .group_by("in_reply_to_user_id")
    .agg(pl.len().alias("in_degree"))
    .rename({"in_reply_to_user_id": "user_id"})
)

replies_sent = (
    internal_replies
    .group_by("author_id")
    .agg(pl.len().alias("out_degree"))
    .rename({"author_id": "user_id"})
)

# Create node dataframe
internal_nodes = pl.DataFrame({"user_id": list(internal_user_ids)})

internal_nodes = (
    internal_nodes
    .join(replies_received, on="user_id", how="left")
    .join(replies_sent, on="user_id", how="left")
    .with_columns([
        pl.col("in_degree").fill_null(0),
        pl.col("out_degree").fill_null(0)
    ])
    .with_columns(
        (pl.col("in_degree") + pl.col("out_degree")).alias("total_degree")
    )
    .sort("total_degree", descending=True)
)

internal_edges.write_csv(join(src, "Austria-Panel-Network-Edges-Internal.csv"))
internal_nodes.write_csv(join(src, "Austria-Panel-Network-Nodes-Internal.csv"))

print(f"Internal network saved with {len(internal_edges)} edges and {len(internal_nodes)} nodes.")
