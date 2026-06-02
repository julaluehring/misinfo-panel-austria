# ===== SCRIPT OVERVIEW =====
# Task:   Filter the author list to the analysis sample using two criteria:
#         (1) remove authors with >=20 tweets on any single day (bot/spam filter);
#         (2) keep only authors with 50 < followers_count < 100,000.
#         Joins filtered IDs with Austria-Panel-Users.csv.gz to attach user metadata.
# Input:  Austria-Panel-Tweets.csv.gz
#         Austria-Panel-Users.csv.gz
# Output: authors_filtered.csv
# Usage:  python 6-filter_authors.py <DATA_DIR>

import polars as pl
from os.path import join
import sys

src = sys.argv[1] # /misinfo-panel-austria/data

# tweets: get per-author, per-day counts
authors = (
    pl.scan_csv(
        join(src, "Austria-Panel-Tweets.csv.gz"),
        separator=",",
        quote_char='"',
        schema_overrides={"author_id": pl.Utf8, "created_at": pl.Utf8}
    )
    .select(["author_id", "created_at"])
    .with_columns(
        pl.col("created_at")
        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ")
    )
    .with_columns(
        pl.col("created_at").dt.strftime("%Y-%m-%d").alias("day")
    )
)

authors_unique = (
    authors
    .select("author_id")
    .unique()
    .with_columns(pl.col("author_id").cast(pl.Utf8))
).collect()

print(f"Unique authors before filtering: {len(authors_unique):,}")

tweets_per_author = (
    authors
    .group_by(["author_id", "day"])
    .agg(pl.len().alias("tweet_count"))
).collect()

# (1) remove authors with >=20 tweets on any single day
tweets_filtered = tweets_per_author.filter(pl.col("tweet_count") < 20)

authors_activity_filtered = (
    authors_unique
    .join(tweets_filtered, on="author_id", how="inner")
    .select("author_id")
    .unique()
)
print(f"Authors after activity filter (<20 tweets/day): {len(authors_activity_filtered):,}")

# join with user metadata from Austria-Panel-Users.csv.gz
authors_df = (
    pl.read_csv(
        join(src, "Austria-Panel-Users.csv.gz"),
        separator=",",
        quote_char='"',
        schema_overrides={"author_id": pl.Utf8}
    )
    .sort("file")
    .unique(subset=["author_id"], keep="last")
    .join(authors_activity_filtered, on="author_id", how="inner")
)

# (2) follower count filter: 50 < followers < 100,000
follower_subset = authors_df.filter(
    (pl.col("followers_count") > 50) & (pl.col("followers_count") < 100_000)
)
print(f"Authors after follower filter (50 < followers < 100k): {len(follower_subset):,}")

follower_subset.write_csv(join(src, "authors_filtered.csv"))
print(f"Saved to {join(src, 'authors_filtered.csv')}")
