# ===== SCRIPT OVERVIEW =====
# Task:   Compute user-level emotion dynamics features and news-sharing engagement metrics.
#         Features: baseline, variability, instability (MSSD), inertia (exp. decay φ)
#                   for anger and fear; plus n_news_shared, n_untrustworthy_shared,
#                   n_biased_shared.
#         Weeks with <10 tweets excluded from instability/inertia. Null engagement
#         counts filled with 0.
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/emotions/emotion_inference.csv.gz
#         <DATA_DIR>/authors_filtered.csv
# Output: <DATA_DIR>/Austria-Panel-Users-Features-Tweet.csv
#         <DATA_DIR>/Austria-Panel-Users-Daily.csv.gz
# Usage:  python 4-extract_features.py <DATA_DIR> [n_test_users]

import sys
from os.path import join
import polars as pl
import numpy as np

src = sys.argv[1]
dst = "../../data"
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz" 
n_test_users = int(sys.argv[2]) if len(sys.argv) > 2 else None # test mode

''' 1) DATA PROCESSING '''
def process_data(src, file):
    df = (pl.scan_csv(join(src, file),
                    separator=",",
                    quote_char='"')
            .select([
                "id",
                "author_id",
                "created_at", 
                "ng_domain",
                "Score",
                "Orientation"])
            .with_columns([
                pl.col("id").cast(pl.Utf8),
                pl.col("author_id").cast(pl.Utf8),
                pl.col("created_at").str.strptime(
                    pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ"),
                pl.col("Score").cast(pl.Float64)])
            .with_columns([
                pl.col("created_at").dt.date().alias("day")
            ])
        )

    authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                    separator=",",
                    quote_char='"')
                .select(pl.col("author_id").cast(pl.Utf8))
                .unique()
            )

    if n_test_users is not None:
        print(f"  TEST MODE: Sampling {n_test_users} users")
        authors = (
            authors.collect()
                .sample(n=n_test_users, seed=63)
                .lazy()
        )


    df_filtered = (df
                    .join(authors, on="author_id", how="inner")
                    .filter(pl.col("day") >= pl.date(2019, 1, 1))
                    .filter(pl.col("day") < pl.date(2023, 4, 1))
    )

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

    df_emotions = (
        df_filtered.join(emotions, left_on="id", right_on="id", how="left")
        .with_columns([
            pl.when(pl.col("ng_domain").is_not_null()).then(1).otherwise(0).alias("is_news"),
            pl.when(pl.col("Score") < 60).then(1).otherwise(0).alias("is_untrustworthy"),
            pl.when(
                pl.col("Orientation").is_in([
                    "Left", "Far Left", "Slightly Left",
                    "Right", "Far Right", "Slightly Right"])
            ).then(1).otherwise(0).alias("is_biased"),
            pl.col("day").dt.strftime("%G-%V").alias("year_week")  # ISO week
        ])
        # sort by author and time
        .sort(["author_id", "created_at"])
    )

    return df_emotions.collect() # lazy evaluation


''' 2) FEATURE EXTRACTION '''
def extract_features(df_emotions: pl.DataFrame, 
                                min_tweets_per_week=10) -> pl.DataFrame:
    """
    Feature extraction using vectorized operations
    """
    
    # calculate baseline per user across whole observation period
    baselines = (
        df_emotions
        .group_by("author_id")
        .agg([
            pl.col("anger_v2").mean().alias("anger_baseline"),
            pl.col("fear_v2").mean().alias("fear_baseline")
        ])
    )

    # calculate variability per user across whole observation period
    variability = (
        df_emotions
        .group_by("author_id")
        .agg([
            pl.col("anger_v2").std().alias("anger_variability"),
            pl.col("fear_v2").std().alias("fear_variability"),
        ])
    )

    # filter to weeks with sufficient tweets
    weekly_counts = (
        df_emotions
        .group_by(['author_id', 'year_week'])
        .agg(pl.len().alias('n_tweets'))
        .filter(pl.col('n_tweets') >= min_tweets_per_week)
    )

    df_filtered = df_emotions.join(
        weekly_counts.select(['author_id', 'year_week']), 
        on=['author_id', 'year_week'], 
        how='inner'
    )

    # calculate within-week MSSD with proper time filtering
    mssd_data = (
        df_filtered
        .sort(["author_id", "year_week", "created_at"])
        .with_columns([
            # get previous values within same author-week with .shift + .over
            pl.col("anger_v2").shift(1).over(["author_id", "year_week"]).alias("anger_prev"),
            pl.col("fear_v2").shift(1).over(["author_id", "year_week"]).alias("fear_prev"),
            pl.col("created_at").shift(1).over(["author_id", "year_week"]).alias("created_at_prev"),
        ])
        .with_columns([
            # calculate time differences in days
            ((pl.col("created_at") - pl.col("created_at_prev")).dt.total_seconds() / (24 * 3600)).alias("delta_days"),
        ])
        # filter to valid time differences (>0 days) and non-missing previous values
        .filter(
            (pl.col("delta_days") > 0) &
            (pl.col("anger_prev").is_not_null()) &
            (pl.col("fear_prev").is_not_null())
        )
        .with_columns([
            # calculate squared differences
            ((pl.col("anger_v2") - pl.col("anger_prev")) ** 2).alias("anger_sqdiff"),
            ((pl.col("fear_v2") - pl.col("fear_prev")) ** 2).alias("fear_sqdiff"),
        ])
    )
    
    # average within weeks, then across weeks per user
    instability = (
        mssd_data
        .group_by(["author_id", "year_week"])
        .agg([
            pl.col("anger_sqdiff").mean().alias("anger_mssd_week"),
            pl.col("fear_sqdiff").mean().alias("fear_mssd_week"),
        ])
        .group_by("author_id")
        .agg([
            pl.col("anger_mssd_week").mean().alias("anger_instability"),
            pl.col("fear_mssd_week").mean().alias("fear_instability"),
        ])
    )

    # calculate inertia using vectorized exponential decay fitting
    def calculate_inertia(anger_vals, fear_vals, timestamps):
        """
        returns φ_hour: emotion persistence per hour
        """
        results = {'anger': np.nan, 'fear': np.nan}
        
        for emotion, values in [('anger', anger_vals), ('fear', fear_vals)]:
            if len(values) < 2:
                continue
                
            vals = np.array(values, dtype=float) # ensure float for NaN handling
            times = np.array(timestamps)
            
            x_t = vals[:-1]
            x_t1 = vals[1:]
            
            # Delta in HOURS
            delta_t = (times[1:] - times[:-1]).astype('timedelta64[h]').astype(float)
            
            # filter: positive and non-zero only
            mask = (
                (delta_t > 0) & 
                (x_t != 0) & (x_t1 != 0) &
                (~np.isnan(x_t.astype(float))) &  
                (~np.isnan(x_t1.astype(float)))   
            )
        
            if mask.sum() < 2: # not enough data points to fit
                continue
            
            x_t = x_t[mask]
            x_t1 = x_t1[mask]
            delta_t = delta_t[mask]
            
            if np.std(x_t) == 0 or np.std(x_t1) == 0: # constant values
                continue
            
            y = np.log(np.abs(x_t1 / x_t)) # log ratio (can be pos or neg)
            X = -delta_t # negative time difference delta_t
            lambda_hat = np.sum(X * y) / np.sum(X * X) # OLS estimate
            persistence_rate = np.abs(lambda_hat) # only positive rates 
            phi = np.exp(-persistence_rate) # decay factor, always between 0 and 1

            if phi > 1.0:
                print(f"\n PHI > 1 DETECTED for {emotion}:")
                print(f"  lambda_hat = {lambda_hat}")
                print(f"  abs(lambda_hat) = {persistence_rate}")
                print(f"  exp(-abs(lambda_hat)) = {phi}")
                print(f"  This should be impossible!")

            results[emotion] = phi  # φ_hour
        
        return results
    
    # apply per week
    weekly_inertia_list = []
    
    for (author_id, year_week), group_df in df_filtered.group_by(['author_id', 'year_week']):
        inertia_results = calculate_inertia(
            group_df['anger_v2'].to_list(),
            group_df['fear_v2'].to_list(),
            group_df['created_at'].to_list()
        )
        
        weekly_inertia_list.append({
            'author_id': author_id,
            'year_week': year_week,
            'anger_inertia_week': inertia_results['anger'],
            'fear_inertia_week': inertia_results['fear']
        })
    
    weekly_inertia = pl.DataFrame(weekly_inertia_list)
    
    # average across weeks per user
    inertia = (
        weekly_inertia
        .group_by('author_id')
        .agg([
            pl.col('anger_inertia_week').drop_nans().mean().alias('anger_inertia'),
            pl.col('fear_inertia_week').drop_nans().mean().alias('fear_inertia')
        ])
    )

    # count the number of times a user has engaged with news
    engagement_counts = (
        df_emotions
        .filter(pl.col("is_news") == 1)
        .group_by("author_id")
        .agg([
            pl.col("is_news").count().alias("n_news_shared"),
            pl.col("is_untrustworthy").sum().alias("n_untrustworthy_shared"),
            pl.col("is_biased").sum().alias("n_biased_shared")
        ])
    )

    features_df = (
        baselines
        .join(variability, on='author_id', how='left')
        .join(instability, on='author_id', how='left')
        .join(inertia, on='author_id', how='left')
        .join(engagement_counts, on='author_id', how='left')
        .with_columns([
            pl.col("n_news_shared").fill_null(0),
            pl.col("n_untrustworthy_shared").fill_null(0),
            pl.col("n_biased_shared").fill_null(0)
        ])
    )

    return features_df


''' 3) DAILY AGGREGATION '''
def aggregate_daily(df_emotions: pl.DataFrame) -> pl.DataFrame:
    anger_threshold = df_emotions["anger_v2"].quantile(0.75)
    fear_threshold  = df_emotions["fear_v2"].quantile(0.75)
    print(f"  Anger threshold (Q3): {anger_threshold:.4f}")
    print(f"  Fear threshold  (Q3): {fear_threshold:.4f}")

    daily = (
        df_emotions
        .with_columns([
            pl.when(pl.col("anger_v2") >= anger_threshold).then(1).otherwise(0).alias("is_angry"),
            pl.when(pl.col("fear_v2")  >= fear_threshold) .then(1).otherwise(0).alias("is_fearful"),
            pl.when(pl.col("Score") >= 60).then(1).otherwise(0).alias("is_trustworthy"),
        ])
        .group_by(["author_id", "day"])
        .agg([
            pl.len().alias("n_tweets"),
            pl.col("is_news").sum().alias("n_news"),
            pl.col("is_untrustworthy").sum().alias("n_untrustworthy"),
            pl.col("is_trustworthy").sum().alias("n_trustworthy"),
            pl.col("is_angry").sum().alias("n_anger"),
            pl.col("is_fearful").sum().alias("n_fear"),
            pl.col("is_biased").sum().alias("n_biased"),
        ])
        .with_columns([
            (pl.col("n_news")          / pl.col("n_tweets")).alias("prop_news"),
            (pl.col("n_anger")         / pl.col("n_tweets")).alias("prop_anger"),
            (pl.col("n_fear")          / pl.col("n_tweets")).alias("prop_fear"),
            (pl.col("n_untrustworthy") / pl.col("n_tweets")).alias("prop_untrustworthy"),
            (pl.col("n_trustworthy")   / pl.col("n_tweets")).alias("prop_trustworthy"),
            (pl.col("n_biased")        / pl.col("n_tweets")).alias("prop_biased"),
        ])
        .sort(["author_id", "day"])
    )
    return daily


''' 4) MAIN EXECUTION '''
if __name__ == "__main__":
    import time
    
    start_time = time.time()
    print("=== DATA PROCESSING ===")
    df_emotions = process_data(src, file)
    print(f"Loaded {len(df_emotions):,} tweets")

    print("\n=== FEATURE EXTRACTION ===")
    features_df = extract_features(df_emotions)

    features_df.write_csv(join(dst, "Austria-Panel-Users-Features-Tweet.csv"))
    print(f"Saved features for {features_df.height:,} authors.")

    print("\n=== DAILY AGGREGATION ===")
    daily_df = aggregate_daily(df_emotions)
    daily_df.write_csv(join(dst, "Austria-Panel-Users-Daily.csv.gz"))
    print(f"Saved daily aggregates: {daily_df['author_id'].n_unique():,} authors, {len(daily_df):,} author-day rows.")

    elapsed_time = (time.time() - start_time) / 60
    print(f"\n=== COMPLETED IN {elapsed_time:.2f} MINUTES ===")