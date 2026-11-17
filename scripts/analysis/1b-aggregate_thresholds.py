# Task:   Daily counts of untrustworthy news at NewsGuard cutoffs 40, 50, 60 and 70 (for the
#         robustness checks in 3a).
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Daily-Thresholds.csv
# Run:    python 1b-aggregate_thresholds.py ../../data

import polars as pl
from os.path import join
import sys

src = sys.argv[1]
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"
thresholds = [40, 50, 60, 70]

df = (pl.scan_csv(join(src, file),
                  separator=",",
                  quote_char='"')
        .select([
            "id",
            "author_id",
            "created_at",
            "Score"])
        .with_columns([
            pl.col("author_id").cast(pl.Utf8),
            pl.col("created_at").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ"),
            pl.col("Score").cast(pl.Float64)])
        .with_columns([
            pl.col("created_at").dt.strftime("%Y-%m-%d").alias("day")
        ])
    )

# keep only authors that we subsetted
authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                  separator=",",
                  quote_char='"')
            .select(pl.col("author_id").cast(pl.Utf8))
            .unique()
        )

df_filtered = (df
                .join(authors,
                    on="author_id",
                    how="inner")
                .filter(pl.col("day") >= "2019-01-01")
                .filter(pl.col("day") < "2023-04-01")
)

# untrustworthy = Score below cutoff (tweet-level score, i.e. the mean
# score for tweets linking to several domains, as in the main measure)
daily_thresholds = (df_filtered
    .group_by("day")
    .agg(
        [pl.len().alias("tweet_count"),
         pl.col("Score").is_not_null().sum().alias("n_rated")] +
        [(pl.col("Score") < t).sum().alias(f"N_{t}") for t in thresholds]
    )
    .sort("day")
    .collect()
)

print(daily_thresholds.select(pl.exclude("day").sum()))

daily_thresholds.write_csv(join(src, "Austria-Panel-Daily-Thresholds.csv"))
print(f"Saved data to {join(src, 'Austria-Panel-Daily-Thresholds.csv')}")
