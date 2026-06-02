# ===== SCRIPT OVERVIEW =====
# Task:   For each user, extract tweet windows before and after news-sharing events
#         (NewsGuard-rated domains) to study emotional reactions. Only windows with
#         >= 10 tweets in both before and after periods are retained.
#         Window output stores emotion lists as space-separated strings.
# Input:  <DATA_DIR>/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz
#         <DATA_DIR>/emotions/emotion_inference.csv.gz
#         <DATA_DIR>/authors_filtered.csv
# Output: <DATA_DIR>/Austria-Panel-Windows-{WINDOW_HOURS}h-10min.csv.gz
# Usage:  python 7-aggregate_windows.py <DATA_DIR> <WINDOW_HOURS>

import polars as pl
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import sys
from os.path import join, exists 
import os
import gzip
import shutil
from tqdm import tqdm

src = sys.argv[1]
WINDOW_HOURS = int(sys.argv[2])
output_file = join(src, "Austria-Panel-Tweets-Windows-" + str(WINDOW_HOURS) + "h-10min.csv")
n_users = int(sys.argv[3]) if len(sys.argv) > 4 else None # test mode

''' STEP 1: Load user list '''
ids = pd.read_csv(join(src, "authors_filtered.csv"), usecols=["author_id"])
user_ids = ids["author_id"].astype(int).tolist()  # convert to list, not set

# take a random sample of ids
if n_users is not None:
    user_ids = pd.Series(user_ids)\
                    .sample(n=n_users, random_state=63)\
                    .tolist()

print(f"Processing {len(user_ids):,} users")

USER_BATCH_SIZE = 1000  

''' STEP 2: Process users in batches '''
for batch_idx in tqdm(range(0, len(user_ids), USER_BATCH_SIZE), 
                      desc="User batches",
                      total=(len(user_ids) + USER_BATCH_SIZE - 1) // USER_BATCH_SIZE):
    
    batch_users = user_ids[batch_idx:batch_idx + USER_BATCH_SIZE]
    
    # Load ONLY this batch of users
    print(f"\nBatch {batch_idx // USER_BATCH_SIZE + 1}: Loading {len(batch_users)} users...")
    
    tweets = (
        pl.scan_csv(
            join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"),
            low_memory=True
        )
        .filter(pl.col("author_id").is_in(batch_users))
        .select(["author_id", "id", "created_at", "Score"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("created_at").str.to_datetime(time_zone="UTC")
        ])
        .filter(
            (pl.col("created_at") >= pl.datetime(2019, 1, 1, time_zone="UTC")) &
            (pl.col("created_at") < pl.datetime(2023, 4, 1, time_zone="UTC"))
        )
    )
    
    emotion_scores = (
        pl.scan_csv(join(src, "./emotions/emotion_inference.csv.gz"))
        .select(["id", "anger_v2", "fear_v2"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("anger_v2").cast(pl.Float64),
            pl.col("fear_v2").cast(pl.Float64)
        ])
    )
    
    tweets_with_emotions = (
        tweets
        .join(emotion_scores, on="id", how="left")
        .collect(streaming=True)
        .sort(["author_id", "created_at"])
    )
    
    # get news events for this batch
    news_tweets = (
        tweets_with_emotions
        .filter(pl.col("Score").is_not_null())
        .unique(subset=["id"])
        .select([
            "id", "author_id", "created_at", "Score",
            "anger_v2", "fear_v2"
        ])
        .rename({
            "id": "event_id",
            "created_at": "event_timestamp",
            "Score": "event_score"
        })
    )
    
    if len(news_tweets) == 0:
        print("  No news events in this batch, skipping...")
        continue
    
    print(f"  Processing {len(news_tweets):,} events...")
    
    # Partition by user
    user_groups = tweets_with_emotions.partition_by("author_id", as_dict=True)
    user_events = news_tweets.partition_by("author_id", as_dict=True)
    
    results_list = []
    
    for author_id_tuple in user_events.keys():
        
        #check if exists in both dicts
        if author_id_tuple not in user_groups:
            continue
        
        user_tweets = user_groups[author_id_tuple]
        user_news = user_events[author_id_tuple]

        author_id = author_id_tuple[0] # get integer
        
        # Process each news event
        for event_row in user_news.iter_rows(named=True):
            event_id = event_row["event_id"]
            event_time = event_row["event_timestamp"]
            event_score = event_row["event_score"]
            
            # Define window
            window_start = event_time - pl.duration(hours=WINDOW_HOURS)
            window_end = event_time + pl.duration(hours=WINDOW_HOURS)
            
            # Filter tweets in window
            window_tweets = user_tweets.filter(
                (pl.col("created_at") >= window_start) &
                (pl.col("created_at") <= window_end) &
                (pl.col("id") != event_id)
            )
            
            if len(window_tweets) == 0:
                continue
            
            # split into before/after
            before_tweets = window_tweets.filter(pl.col("created_at") < event_time)
            after_tweets = window_tweets.filter(pl.col("created_at") > event_time)
            
            # process the emotion scores
            def process_emotions(df):
                
                n_tweets = len(df)
                if n_tweets == 0:
                    return None

                # create lists of emotion scores
                anger_list = df["anger_v2"].drop_nulls().to_list()
                fear_list = df["fear_v2"].drop_nulls().to_list()

                # check if valid scores 
                if len(anger_list) == 0 or len(fear_list) == 0:
                    return None

                # only calculate mean if enough tweets
                mean_anger = df["anger_v2"].mean() if len(anger_list) >= 10 else float('nan')

                mean_fear = df["fear_v2"].mean() if len(fear_list) >= 10 else float('nan')

                return {
                    "n_tweets": n_tweets,
                    "n_anger": len(anger_list), 
                    "n_fear": len(fear_list),
                    "anger_list": anger_list,
                    "fear_list": fear_list,
                    "mean_anger": mean_anger,
                    "mean_fear": mean_fear
                }
            
            before_stats = process_emotions(before_tweets)
            after_stats = process_emotions(after_tweets)

            # only keep if one windows has anger or fear
            if before_stats and after_stats:
                results_list.append({
                    "event_id": event_id,
                    "author_id": author_id,
                    "event_timestamp": event_time,
                    "event_score": event_score,
                    "n_tweets_before": before_stats["n_tweets"],
                    "n_anger_before": before_stats["n_anger"],
                    "n_fear_before": before_stats["n_fear"],
                    "anger_before_list": " ".join(map(str, before_stats["anger_list"])),  
                    "fear_before_list": " ".join(map(str, before_stats["fear_list"])), 
                    "mean_anger_before": before_stats["mean_anger"],
                    "mean_fear_before": before_stats["mean_fear"],
                    "n_tweets_after": after_stats["n_tweets"],
                    "n_anger_after": after_stats["n_anger"],
                    "n_fear_after": after_stats["n_fear"],
                    "anger_after_list": " ".join(map(str, after_stats["anger_list"])),
                    "fear_after_list": " ".join(map(str, after_stats["fear_list"])),
                    "mean_anger_after": after_stats["mean_anger"],
                    "mean_fear_after": after_stats["mean_fear"],
                    "event_anger": event_row.get("anger_v2"),
                    "event_fear": event_row.get("fear_v2"),
                })
    
    
    # how many results to write???
    print(f"  Preparing to write {len(results_list):,} results...")

    if results_list:
        df_batch = pl.DataFrame(results_list, infer_schema_length=None)
        
        if not exists(output_file):  
            df_batch.write_csv(output_file)
            print(f"  Wrote {len(df_batch):,} rows (new file)")
        else:
            with open(output_file, 'ab') as f:
                df_batch.write_csv(f, include_header=False)
            print(f"  Appended {len(df_batch):,} rows")
    
    # Free memory before next batch
    del tweets_with_emotions, news_tweets, user_groups, user_events, results_list

''' STEP 3: Compress '''
print("\nCompressing final results...")
with open(output_file, 'rb') as f_in:
    with gzip.open(output_file + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(output_file)

print("Done!")