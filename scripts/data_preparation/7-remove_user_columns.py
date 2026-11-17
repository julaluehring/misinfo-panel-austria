# Task:   Remove user columns (username, follower counts, location, ...) from the tweet file.
# Input:  data/Austria-Panel-Tweets.csv.gz (not released)
#         data/dtypes.pkl
# Output: data/Austria-Panel-Tweets.csv.gz (overwritten)
# Run:    python 7-remove_user_columns.py ../../data

import pandas as pd
import sys
from os.path import join
import pickle as pkl
import gzip
import os
import tempfile
from tqdm import tqdm

src = sys.argv[1] # <DATA_DIR>
file = "Austria-Panel-Tweets.csv.gz"
out = join(src,"Austria-Panel-Tweets.csv.gz")

dtypes = pkl.load(open(
    join(src, "dtypes.pkl"), "rb"))
chunk_size = 20_000_000  

columns_to_drop = ["username", "verified", 
                   "followers_count", "following_count",
                   "tweet_count", "location", "geo_location"]

# Estimated total rows for tqdm progress bar
total_rows = 206_000_000
total_chunks = total_rows // chunk_size + 1

tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv.gz", dir=src)
os.close(tmp_fd)

try:
    with gzip.open(tmp_path, 'wt') as f_out:
        for i, chunk in enumerate(tqdm(pd.read_csv(join(src, file),
                                                   compression="gzip",
                                                   dtype=dtypes,
                                                   low_memory=False,
                                                   chunksize=chunk_size),
                                       total=total_chunks,
                                       desc="Processing chunks")):
            chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
            chunk.to_csv(f_out, index=False, header=(i == 0))
    os.replace(tmp_path, out)
except Exception:
    os.remove(tmp_path)
    raise

print("Done!")