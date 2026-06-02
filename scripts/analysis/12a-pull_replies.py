# ===== SCRIPT OVERVIEW =====
# Task:   Build the full reply network from the tweet dataset.
#         Classifies edges as internal (both users in sample), outgoing, or incoming.
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/Austria-Panel-Users-Daily.csv.gz
# Output: <DATA_DIR>/Austria-Panel-Network-Edges.csv.gz
#         <DATA_DIR>/Austria-Panel-Network-Nodes.csv.gz
#         <DATA_DIR>/users/reply_analysis_summary.csv
#         <DATA_DIR>/users/top_replied_to_users.csv
# Usage:  python 12a-pull_replies.py <DATA_DIR>

import polars as pl
import sys
from os.path import join
import os

src = sys.argv[1]

# get the author ids of users that are in our sample
authors = pl.read_csv(join(
    src, "Austria-Panel-Users-Daily.csv.gz"),
                        columns=["author_id"]
                       )["author_id"].unique().to_list()

authors_set = set(authors)

# from our tweet sample, get the ids of replied to authors
# and limit to observation period
replied_to_df = (
    pl.scan_csv(join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"),
        low_memory=True)
        .select(["in_reply_to_user_id", "created_at", "author_id"])
        .filter(pl.col("in_reply_to_user_id").is_not_null())
        .with_columns(pl.col("created_at").str.to_datetime(time_zone="UTC"))
        .filter(pl.col("created_at") >= pl.datetime(2019, 1, 1, time_zone="UTC"))
        .filter(pl.col("created_at") < pl.datetime(2023, 4, 1, time_zone="UTC"))
).collect()

# keep only the ones that are replies
replied_to_df = replied_to_df.filter(
    pl.col("in_reply_to_user_id").is_not_null())

# who are the most replied to users?
(replied_to_df
    .group_by("in_reply_to_user_id")
    .agg(pl.len().alias("reply_count"))
    .sort("reply_count", descending=True)
    .head(20)
    .write_csv(join("users", "top_replied_to_users.csv"))
)

# whats the overlap? how many do we have in our sample?
unique_replied_to = replied_to_df["in_reply_to_user_id"]\
                        .unique().to_list()
unique_replied_to_set = set(unique_replied_to)

overlap = authors_set.intersection(unique_replied_to_set)
overlap_count = len(overlap)
total_replied_to = len(unique_replied_to)
replies_in_sample = replied_to_df.filter(
    pl.col("in_reply_to_user_id").is_in(authors)
).height
replies_outside_sample = len(replied_to_df) - replies_in_sample

(pl.DataFrame({
    "Metric": [
        "Unique users replied to",
        "Users replied to IN sample",
        "Users replied to OUTSIDE sample",
        "Overlap percentage",
        "Replies to users IN sample",
        "Replies to users OUTSIDE sample"
    ],
    "Value": [
        total_replied_to,
        overlap_count,
        total_replied_to - overlap_count,
        None,
        replies_in_sample,
        replies_outside_sample
    ],
    "Percentage": [
        None,
        None,
        None,
        f"{overlap_count / total_replied_to * 100:.2f}%",
        None,
        None
    ]})
    .write_csv(join("users", "reply_analysis_summary.csv"))
)

''' CREATE NETWORK DATA FOR VISUALIZATION '''
print("\nCreating reply network data...")

edges = (
    replied_to_df
    .group_by(["author_id", "in_reply_to_user_id"])
    .agg(pl.len().alias("weight"))
    .with_columns([
        pl.col("author_id").is_in(authors).alias("source_in_sample"),
        pl.col("in_reply_to_user_id").is_in(authors).alias("target_in_sample")
    ])
    .with_columns(
        pl.when(pl.col("source_in_sample") & pl.col("target_in_sample"))
        .then(pl.lit("internal"))
        .when(~pl.col("target_in_sample"))
        .then(pl.lit("outgoing"))
        .otherwise(pl.lit("incoming"))
        .alias("edge_type")
    )
)

edges.write_csv(join(src, "Austria-Panel-Network-Edges.csv.gz"))
print(f"Edges saved: {len(edges)} unique connections")

all_users = set(replied_to_df["author_id"].unique().to_list()) | \
            set(replied_to_df["in_reply_to_user_id"].unique().to_list())

replies_received = (
    replied_to_df
    .group_by("in_reply_to_user_id")
    .agg(pl.len().alias("n_replies_received"))
    .rename({"in_reply_to_user_id": "user_id"})
)

replies_sent = (
    replied_to_df
    .group_by("author_id")
    .agg(pl.len().alias("n_replies_sent"))
    .rename({"author_id": "user_id"})
)

nodes = pl.DataFrame({"user_id": list(all_users)})

nodes = (
    nodes
    .join(replies_received, on="user_id", how="left")
    .join(replies_sent, on="user_id", how="left")
    .with_columns([
        pl.col("n_replies_received").fill_null(0),
        pl.col("n_replies_sent").fill_null(0)
    ])
    .with_columns([
        (pl.col("n_replies_received") + pl.col("n_replies_sent")).alias("total_activity"),
        pl.col("user_id").is_in(authors).alias("in_sample")
    ])
    .with_columns(
        pl.when(pl.col("in_sample"))
        .then(pl.lit("inside"))
        .otherwise(pl.lit("outside"))
        .alias("sample_status")
    )
    .sort("total_activity", descending=True)
)

nodes.write_csv(join(src, "Austria-Panel-Network-Nodes.csv.gz"))
print(f"Nodes saved: {len(nodes)} unique users")
