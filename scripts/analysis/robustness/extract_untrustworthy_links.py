# Task:   Links to untrustworthy sources (NewsGuard < 60), for newsguard_robustness.Rmd.
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Untrustworthy-Totals.csv
#         data/Austria-Panel-Untrustworthy-Links.csv (not released)
# Run:    python extract_untrustworthy_links.py ../../../data

import polars as pl
from os.path import join
import sys


src = sys.argv[1]
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"

df = (pl.scan_csv(join(src, file),
                  separator=",",
                  quote_char='"')
        .select([
            "id",
            "author_id",
            "created_at",
            "ng_domain",
            "Score"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("author_id").cast(pl.Utf8),
            pl.col("created_at").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ"),
            pl.col("Score").cast(pl.Float64)])
        .with_columns([
            pl.col("created_at").dt.strftime("%Y-%m-%d").alias("day"),
            pl.col("created_at").dt.strftime("%Y-%m").alias("month")
        ])
    )

# keep only authors that we subsetted
authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                  separator=",",
                  quote_char='"')
            .select(pl.col("author_id").cast(pl.Utf8))
            .unique()
        )

# same panel and observation period as 1-aggregate_data.py
df_filtered = (df
                .join(authors,
                    on="author_id",
                    how="inner")
                .filter(pl.col("day") >= "2019-01-01")
                .filter(pl.col("day") < "2023-04-01")
)

# denominators for reporting
totals = (df_filtered
          .select([
              pl.len().alias("n_tweets"),
              pl.col("Score").is_not_null().sum().alias("n_rated_tweets"),
              (pl.col("Score") < 60).sum().alias("n_untrustworthy_tweets")
          ])
          .collect())
print(totals)
totals.write_csv(join(src, "Austria-Panel-Untrustworthy-Totals.csv"))

# untrustworthy tweets (tweet-level Score < 60, as in the daily measure),
# one row per tweet x domain (multi-domain tweets are joined with ", ")
untrustworthy = (df_filtered
                 .filter(pl.col("Score") < 60)
                 .with_columns(pl.col("ng_domain").str.split(", "))
                 .explode("ng_domain")
                 .with_columns(pl.col("ng_domain").str.strip_chars())
                 .select(["id", "author_id", "day", "month",
                          "ng_domain", "Score"])
                 .collect())

print(f"Untrustworthy tweet-domain rows: {untrustworthy.height}")
print(f"Unique untrustworthy tweets: {untrustworthy['id'].n_unique()}")

untrustworthy.write_csv(join(src, "Austria-Panel-Untrustworthy-Links.csv"))
print(f"Saved data to {join(src, 'Austria-Panel-Untrustworthy-Links.csv')}")
