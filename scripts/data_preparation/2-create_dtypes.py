# ===== SCRIPT OVERVIEW =====
# Task:   Generate dtypes.pkl: a column-type dictionary required by all downstream concat and merge scripts for consistent dtypes.
# Output: <DATA_DIR>/dtypes.pkl
# Usage:  python 2-create_dtypes.py <DATA_DIR>

import pickle as pl
import sys
from os.path import join

src = sys.argv[1] # /misinfo-panel-austria/data

dtypes = {
    "id":str, 
    "conversation_id":str,
    "text":str,
    "lang":str,
    "created_at":str,
    "author_id":str,
    "followers_count":float,
    "following_count":float,
    "tweet_count":float,
    "location":str,
    "geo_location":str,
    "retweet_count":float,
    "reply_count":float,
    "quote_count":float,
    "like_count":float,
    "impression_count":float,
    "urls":str,
    "expanded_urls":str,
    "hashtags":str,
    "mentions":str,
    "tweet_type":str,
    "retweeted_id":str,
    "quoted_id":str,
    "in_reply_to_user_id":str,
    "file":str,
    "anger_v2":float,
    "joy_v2":float,
    "fear_v2":float,
    "sadness_v2":float,
    "disgust_v2":float,
    "enthusiasm_v2":float,
    "pride_v2":float,
    "hope_v2":float,
    "domain":str
    }

# save as pickle file
try:
    with open(join(src, "dtypes.pkl"), "wb") as f:
        pl.dump(dtypes, f)
    print(f"File saved at {join(src, 'dtypes.pkl')}")
except Exception as e:
    print(f"An error occurred: {e}")