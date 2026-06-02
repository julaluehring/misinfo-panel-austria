# ===== SCRIPT OVERVIEW =====
# Task:   Compute per-user summary statistics across the full observation period:
#         activity dates, tweet volume, mean trustworthiness, mean emotion scores,
#         and account-level metadata (followers, following, account age).
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/emotions/emotion_inference.csv.gz
#         <DATA_DIR>/authors_filtered.csv
# Output: <DATA_DIR>/Austria-Panel-User-Statistics.csv
# Usage:  python 10-aggregate_user_stats.py <DATA_DIR>

import polars as pl
import pandas as pd
import sys
from os.path import join
from datetime import datetime
import os 

src = sys.argv[1]

print("Loading and processing user data...")

def aggregate_user_stats(src):
    """
    Aggregate user statistics including:
    a) When they joined Twitter
    b) When they started posting (first tweet in our dataset)
    c) How active they are on average (tweets per day)
    d) Their average trustworthiness score
    e) Their baseline anger
    f) Their baseline fear
    """
    
    # Load main tweets data
    print("Loading tweets...")
    tweets = (
        pl.scan_csv(join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"))  # add tweet-level
        .select([
            "id",
            "author_id", 
            "created_at",
            "Score",
            "Orientation",
            "ng_domain",
        ])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("author_id").cast(pl.Utf8),
            pl.col("created_at").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ").alias("tweet_created"),
            pl.col("Score").cast(pl.Float64)
        ])
        .filter(pl.col("tweet_created") >= pl.datetime(2019, 1, 1))
        .filter(pl.col("tweet_created") < pl.datetime(2023, 4, 1))
    )
    
    # Load filtered users list
    print("Loading filtered users...")
    filtered_users = (
        pl.scan_csv(join(src, "authors_filtered.csv"))
        .select("author_id")
        .with_columns(pl.col("author_id").cast(pl.Utf8))
        .unique()
    )
    
    # Filter tweets to only include filtered users
    tweets = tweets.join(filtered_users, on="author_id", how="inner")
    print("Filtered to only include pre-selected users")
    
    # Aggregate the number of tweets per day per user
    avg_tweets_day = (tweets
                      .group_by(["author_id", 
                                pl.col("tweet_created").dt.date()
                                .alias("day")])
                      .agg(pl.count("id").alias("tweets_per_day"))
                      .group_by("author_id")
                      .agg(pl.col("tweets_per_day")
                           .mean().alias("avg_tweets_per_day")
                           )
    )

    # Load user metadata (account creation dates)
    print("Loading user metadata...")
    users = (
            pl.scan_csv(join(src, "authors_filtered.csv")) # user-level data
            .select([
                "author_id",
                "followers_count",
                "following_count",
                "tweet_count",
                "created_at"
            ])
            .with_columns([
                pl.col("author_id").cast(pl.Utf8),
                pl.col("created_at").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ").alias("account_created")
            ])
        )

    
    # Load emotions data
    print("Loading emotions data...")
    emotions = (
        pl.scan_csv(join(src, "emotions/emotion_inference.csv.gz")) # alao at tweet-level
        .select([
            "id",
            "anger_v2", 
            "fear_v2"
        ])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("anger_v2").cast(pl.Float64).alias("anger"),
            pl.col("fear_v2").cast(pl.Float64).alias("fear"),
        ])
    )
    
    # Join tweets with emotions and calculate user statistics in one go
    print("Joining tweets with emotions and calculating statistics...")
    user_stats = (
        tweets
        .join(emotions, on="id", how="left")
        .join(users, on="author_id", how="left")  # join users early in lazy operation
        .group_by("author_id")
        .agg([
            # first and last tweets dates in dataset
            pl.col("tweet_created").min().alias("first_tweet_date"),
            pl.col("tweet_created").max().alias("last_tweet_date"),
            
            # overall activity metrics
            pl.len().alias("total_tweets"),
            
            # avg trustworthiness metrics
            pl.col("Score").mean().alias("avg_trustworthiness"),
            
            # avg emotion baselines
            pl.col("anger").mean().alias("avg_anger"),
            pl.col("fear").mean().alias("avg_fear"),
            
            # account creation date (take first since it's the same for all tweets from user)
            pl.col("account_created").first().alias("account_created"),
            pl.col("followers_count").first().alias("followers_count"),
            pl.col("following_count").first().alias("following_count"),
            pl.col("tweet_count").first().alias("tweet_count"),
        ])
        .collect(streaming=True)  # streaming collection to reduce memory
    )

    # Add avg tweets per day here
    print("Merging with average tweets per day...")
    avg_tweets_day_df = avg_tweets_day.collect()
    user_stats = user_stats.join(avg_tweets_day_df, on="author_id", how="left")
    
    return user_stats


def main():
    print(f"Starting user aggregation from: {src}")
    
    user_stats = aggregate_user_stats(src)
    
    print(f"Calculated statistics for {len(user_stats):,} users")
    
    summary_stats = user_stats.select([
        pl.col("total_tweets").mean().alias("avg_tweets_per_user"),
        pl.col("avg_tweets_per_day").mean().alias("avg_tweets_per_day"),
        pl.col("avg_trustworthiness").mean().alias("mean_trustworthiness"),
        pl.col("avg_anger").mean().alias("mean_baseline_anger"),
        pl.col("avg_fear").mean().alias("mean_baseline_fear"),
        pl.len().alias("n_users")
    ])
    

    output_file = join(src, "Austria-Panel-User-Statistics.csv")
    user_stats.write_csv(output_file)
    print(f"\nSaved user statistics to: {output_file}")

    os.makedirs("./cohorts", exist_ok=True)
    summary_stats.write_csv("./cohorts/user_statistics_summary.csv")
    print(f"Saved summary statistics to: ./cohorts/user_statistics_summary.csv")


if __name__ == "__main__":
    main()