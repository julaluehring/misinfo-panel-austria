# ===== SCRIPT OVERVIEW =====
# Task:   Aggregate tweet-level data (emotions + trustworthiness scores) to a daily time series. 
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (created from inference/trustworthiness/8-match_rating.py)
#         <DATA_DIR>/emotions/emotion_inference.csv.gz
#         <DATA_DIR>/authors_filtered.csv
# Output: <DATA_DIR>/Austria-Panel-Daily.csv
#         Columns: tweet_count, count_ng_domains, trustworthiness bins, mean_score, count_anger, count_fear, mean_anger, mean_fear, count_left, count_right, count_neutral, n_authors
# Usage:  python 1_aggregate_data.py <DATA_DIR>

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
            "Orientation",
            "Score"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
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

# keep only tweets from filtered authors 
# within the observation perios
df_filtered = (df
                .join(authors,
                    on="author_id",
                    how="inner")
                .filter(pl.col("day") >= "2019-01-01")
                .filter(pl.col("day") < "2023-04-01")
)

# add the emotion scores
emotions = (
        pl.scan_csv(
            join(src, "emotions", "emotion_inference.csv.gz"),
                  separator=",")
            .select([
                "id",
                "anger_v2",
                "fear_v2"])
            .with_columns([
                pl.col("id").cast(pl.Utf8),
                pl.col("anger_v2").cast(pl.Float64),
                pl.col("fear_v2").cast(pl.Float64)
            ])
        )

df_emotions = (df_filtered
               .join(emotions, 
                    left_on="id", 
                    right_on="id", 
                    how="left") # keep all rows from df1
    )

# use upper quantile as anger cutoff
anger_quantile = df_emotions.select(
    pl.col("anger_v2").quantile(0.75)
).collect().to_series()[0]
fear_quantile = df_emotions.select(
    pl.col("fear_v2").quantile(0.75)
).collect().to_series()[0]

# aggregate the data per day
daily_summary = (df_emotions
    .group_by("day")
    .agg([
        pl.len().alias("tweet_count"),

        pl.col("ng_domain")
            .is_not_null()
            .sum()
            .alias("count_ng_domains"),

        pl.when(pl.col("Score") <= 39)
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_max_caution"),  # 0–39

        pl.when((pl.col("Score") >= 40) & (pl.col("Score") <= 59))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_caution"),  # 40–59

        pl.when((pl.col("Score") >= 60) & (pl.col("Score") <= 74))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_exceptions"),  # 60–74

        pl.when((pl.col("Score") >= 75) & (pl.col("Score") <= 99))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_generally_credible"),  # 75–99

        pl.when(pl.col("Score") == 100)
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_high_credibility"),  # 100 only

        pl.col("Score").mean().alias("mean_score"),

        pl.col("anger_v2").mean().alias("mean_anger"),

        # angry tweet = anger_v2 >= anger_quantile
        pl.when(pl.col("anger_v2") >= anger_quantile)
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_anger"),
        
        pl.when(pl.col("fear_v2") >= fear_quantile)
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_fear"),

        pl.when(pl.col("Orientation") == "Right")
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_right"),

        pl.when(pl.col("Orientation") == "Neutral")
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_neutral"),   
        
        pl.when(pl.col("Orientation") == "Left")
            .then(1)
            .otherwise(0)
            .sum()
            .alias("count_left"),

        pl.col("fear_v2").mean().alias("mean_fear"),

        pl.col("author_id").n_unique().alias("n_authors")
    ])
    .collect() # <--- only collect now!
)

# save the daily summary
daily_summary.write_csv(
                join(src, "Austria-Panel-Daily.csv"),
                        separator=",",
                        quote_char='"')