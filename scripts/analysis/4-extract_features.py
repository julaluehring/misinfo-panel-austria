# Task:   Per-user emotion features (baseline, variability, instability, inertia), news sharing
#         counts, and daily per-user counts.
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/emotions/emotion_inference.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Users-Features-Tweet-<lang>.csv
#         data/Austria-Panel-Users-Daily.csv.gz (lang = de_en only)
# Run:    python 4-extract_features.py ../../data [lang] [n_test_users]   # lang: de_en (default), de, all

import sys
import gzip
from os.path import join
import polars as pl
import numpy as np

src = sys.argv[1]
dst = src
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz" 
LANG = sys.argv[2] if len(sys.argv) > 2 else "de_en"
LANG_SETS = {"de_en": ["de", "en"], "de": ["de"], "all": None}
assert LANG in LANG_SETS, f"lang must be one of {list(LANG_SETS)}"
n_test_users = int(sys.argv[3]) if len(sys.argv) > 3 else None # test mode
EMOTIONS = ["anger", "fear", "disgust"]
THRESHOLD = 0.65 # emotion presence (main specification)

''' 1) DATA PROCESSING '''
def process_data(src, file):
    df = (pl.scan_csv(join(src, file),
                    separator=",",
                    quote_char='"',
                    schema_overrides={"lang": pl.Utf8})
            .select([
                "id",
                "author_id",
                "created_at", 
                "lang",
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
                    "fear_v2",
                    "disgust_v2"])
                .with_columns([
                    pl.col("id").cast(pl.Utf8),
                    pl.col("anger_v2").cast(pl.Float64),
                    pl.col("fear_v2").cast(pl.Float64),
                    pl.col("disgust_v2").cast(pl.Float64)
                ])
            )

    df_emotions = df_filtered.join(emotions, left_on="id", right_on="id", how="left")

    # language filter: set emotion scores of other languages to missing
    if LANG_SETS[LANG] is not None:
        in_lang = pl.col("lang").is_in(LANG_SETS[LANG]).fill_null(False)
        df_emotions = df_emotions.with_columns([
            pl.when(in_lang).then(pl.col(f"{e}_v2")).otherwise(None).alias(f"{e}_v2")
            for e in EMOTIONS])

    df_emotions = (
        df_emotions
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
            pl.col("fear_v2").mean().alias("fear_baseline"),
            pl.col("disgust_v2").mean().alias("disgust_baseline")
        ])
    )

    # calculate variability per user across whole observation period
    variability = (
        df_emotions
        .group_by("author_id")
        .agg([
            pl.col("anger_v2").std().alias("anger_variability"),
            pl.col("fear_v2").std().alias("fear_variability"),
            pl.col("disgust_v2").std().alias("disgust_variability"),
        ])
    )

    # tweet counts per user (for activity controls)
    activity = (
        df_emotions
        .group_by("author_id")
        .agg([
            pl.len().alias("n_tweets"),
            pl.col("anger_v2").is_not_null().sum().alias("n_scored")
        ])
    )

    # dynamics are computed from consecutive classified tweets only
    df_scored = df_emotions.filter(
        pl.col("anger_v2").is_not_null() &
        pl.col("fear_v2").is_not_null() &
        pl.col("disgust_v2").is_not_null()
    )

    # filter to weeks with sufficient classified tweets
    weekly_counts = (
        df_scored
        .group_by(['author_id', 'year_week'])
        .agg(pl.len().alias('n_tweets'))
        .filter(pl.col('n_tweets') >= min_tweets_per_week)
    )

    df_filtered = df_scored.join(
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
            pl.col("disgust_v2").shift(1).over(["author_id", "year_week"]).alias("disgust_prev"),
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
            (pl.col("fear_prev").is_not_null()) &
            (pl.col("disgust_prev").is_not_null())
        )
        .with_columns([
            # calculate squared differences
            ((pl.col("anger_v2") - pl.col("anger_prev")) ** 2).alias("anger_sqdiff"),
            ((pl.col("fear_v2") - pl.col("fear_prev")) ** 2).alias("fear_sqdiff"),
            ((pl.col("disgust_v2") - pl.col("disgust_prev")) ** 2).alias("disgust_sqdiff"),
        ])
    )
    
    # average within weeks, then across weeks per user
    instability = (
        mssd_data
        .group_by(["author_id", "year_week"])
        .agg([
            pl.col("anger_sqdiff").mean().alias("anger_mssd_week"),
            pl.col("fear_sqdiff").mean().alias("fear_mssd_week"),
            pl.col("disgust_sqdiff").mean().alias("disgust_mssd_week"),
        ])
        .group_by("author_id")
        .agg([
            pl.col("anger_mssd_week").mean().alias("anger_instability"),
            pl.col("fear_mssd_week").mean().alias("fear_instability"),
            pl.col("disgust_mssd_week").mean().alias("disgust_instability"),
        ])
    )

    # calculate inertia using vectorized exponential decay fitting
    def calculate_inertia(emotion_vals, timestamps):
        """
        emotion_vals: dict {emotion: list of scores}
        returns φ_hour: emotion persistence per hour
        """
        results = {emotion: np.nan for emotion in emotion_vals}
        
        for emotion, values in emotion_vals.items():
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
            {'anger': group_df['anger_v2'].to_list(),
             'fear': group_df['fear_v2'].to_list(),
             'disgust': group_df['disgust_v2'].to_list()},
            group_df['created_at'].to_list()
        )
        
        weekly_inertia_list.append({
            'author_id': author_id,
            'year_week': year_week,
            'anger_inertia_week': inertia_results['anger'],
            'fear_inertia_week': inertia_results['fear'],
            'disgust_inertia_week': inertia_results['disgust']
        })
    
    weekly_inertia = pl.DataFrame(weekly_inertia_list)
    
    # average across weeks per user
    inertia = (
        weekly_inertia
        .group_by('author_id')
        .agg([
            pl.col('anger_inertia_week').drop_nans().mean().alias('anger_inertia'),
            pl.col('fear_inertia_week').drop_nans().mean().alias('fear_inertia'),
            pl.col('disgust_inertia_week').drop_nans().mean().alias('disgust_inertia')
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
        .join(activity, on='author_id', how='left')
        .with_columns([
            pl.col("n_news_shared").fill_null(0),
            pl.col("n_untrustworthy_shared").fill_null(0),
            pl.col("n_biased_shared").fill_null(0)
        ])
    )

    return features_df


''' 3) DAILY AGGREGATION '''
def aggregate_daily(df_emotions: pl.DataFrame) -> pl.DataFrame:
    """
    Daily counts per author. Emotion presence: score >= THRESHOLD (0.65) among
    classified tweets in the selected languages; emotion shares use n_scored
    as denominator. News shares use all tweets (n_tweets).
    """
    daily = (
        df_emotions
        .with_columns([
            pl.col("anger_v2").is_not_null().cast(pl.Int64).alias("is_scored"),
            (pl.col("anger_v2") >= THRESHOLD).fill_null(False).cast(pl.Int64).alias("is_angry"),
            (pl.col("fear_v2") >= THRESHOLD).fill_null(False).cast(pl.Int64).alias("is_fearful"),
            (pl.col("disgust_v2") >= THRESHOLD).fill_null(False).cast(pl.Int64).alias("is_disgusted"),
            pl.when(pl.col("Score") >= 60).then(1).otherwise(0).alias("is_trustworthy"),
        ])
        .group_by(["author_id", "day"])
        .agg([
            pl.len().alias("n_tweets"),
            pl.col("is_scored").sum().alias("n_scored"),
            pl.col("is_news").sum().alias("n_news"),
            pl.col("is_untrustworthy").sum().alias("n_untrustworthy"),
            pl.col("is_trustworthy").sum().alias("n_trustworthy"),
            pl.col("is_angry").sum().alias("n_anger"),
            pl.col("is_fearful").sum().alias("n_fear"),
            pl.col("is_disgusted").sum().alias("n_disgust"),
            pl.col("is_biased").sum().alias("n_biased"),
        ])
        .with_columns([
            (pl.col("n_news")          / pl.col("n_tweets")).alias("prop_news"),
            # emotion shares: NaN -> null on days without classified tweets
            pl.when(pl.col("n_scored") > 0).then(pl.col("n_anger")   / pl.col("n_scored")).alias("prop_anger"),
            pl.when(pl.col("n_scored") > 0).then(pl.col("n_fear")    / pl.col("n_scored")).alias("prop_fear"),
            pl.when(pl.col("n_scored") > 0).then(pl.col("n_disgust") / pl.col("n_scored")).alias("prop_disgust"),
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
    print(f"Language set: {LANG}")
    df_emotions = process_data(src, file)
    print(f"Loaded {len(df_emotions):,} tweets "
          f"({df_emotions['anger_v2'].is_not_null().sum():,} classified in {LANG})")

    print("\n=== FEATURE EXTRACTION ===")
    features_df = extract_features(df_emotions)

    features_df.write_csv(join(dst, f"Austria-Panel-Users-Features-Tweet-{LANG}.csv"))
    print(f"Saved features for {features_df.height:,} authors.")

    if LANG == "de_en":
        print("\n=== DAILY AGGREGATION ===")
        daily_df = aggregate_daily(df_emotions)
        with gzip.open(join(dst, "Austria-Panel-Users-Daily.csv.gz"), "wb") as f:
            daily_df.write_csv(f)
        print(f"Saved daily aggregates: {daily_df['author_id'].n_unique():,} authors, "
              f"{len(daily_df):,} author-day rows.")

    elapsed_time = (time.time() - start_time) / 60
    print(f"\n=== COMPLETED IN {elapsed_time:.2f} MINUTES ===")