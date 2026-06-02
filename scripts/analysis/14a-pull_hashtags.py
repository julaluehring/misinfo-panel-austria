# ===== SCRIPT OVERVIEW =====
# Task:   Aggregate tweet-level hashtags to a daily list per observation day.
#         Filters to the analysis sample and observation period before aggregating.
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/authors_filtered.csv
# Output: <DATA_DIR>/Austria-Panel-Daily-Hashtags.csv
# Usage:  python 1b-aggregate_hashtags.py <DATA_DIR>

import polars as pl
from os.path import join
import sys

src = sys.argv[1]
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"

df = (pl.scan_csv(join(src, file),
                  separator=",",
                  quote_char='"')
        .select([
            "created_at",
            "author_id",
            "hashtags"])
        .with_columns([
            pl.col("created_at").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ"),
            pl.col("hashtags").cast(pl.Utf8),
            pl.col("author_id").cast(pl.Utf8)])
        .with_columns([
            pl.col("created_at").dt.strftime("%Y-%m-%d").alias("day")
        ])
    )

authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                  separator=",",
                  quote_char='"')
            .select(pl.col("author_id").cast(pl.Utf8))
            .unique()
        )

df_filtered = (df
                .join(authors, on="author_id", how="inner")
                .filter(pl.col("day") >= "2019-01-01")
                .filter(pl.col("day") < "2023-04-01")
)

daily_hashtags = (df_filtered
                  .select(["day", "hashtags"])
                  .filter(pl.col("hashtags").is_not_null())
                  .filter(pl.col("hashtags") != "")
                  .with_columns([
                      pl.col("hashtags").str.split(",").alias("hashtags")
                  ])
                  .explode("hashtags")
                  .group_by("day")
                  .agg([
                      pl.col("hashtags").str.concat(",").alias("hashtags"),
                      pl.col("hashtags").len().alias("hashtag_count")
                  ])
)

daily_hashtags.collect().write_csv(join(src, "Austria-Panel-Daily-Hashtags.csv"))
print("Saved Austria-Panel-Daily-Hashtags.csv")
