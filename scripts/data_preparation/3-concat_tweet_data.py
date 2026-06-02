# ===== SCRIPT OVERVIEW =====
# Task:   Concatenate all tweet CSVs into a single compressed master file.
# Input:  <DATA_DIR>/CSVtweets/*.csv
#         <DATA_DIR>/dtypes.pkl
# Output: <DATA_DIR>/Austria-Panel-Tweets.csv.gz
# Usage:  python 2-concat_tweet_data.py <DATA_DIR>

import os
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
import sys
import pickle as pl

src = sys.argv[1] # /misinfo-panel-austria/data
input_dir = join(src, "CSVtweets")
output_file = join(src, "Austria-Panel-Tweets.csv.gz")

# extract a list of files in the directory
files = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f))]
# files = files[:3] # test strategy

# define data types
dtypes = pl.load(open(join(src, "dtypes.pkl"), "rb"))

# loop through the files and append to a new csv-file
with tqdm(total=len(files), desc="Processing files", unit="file") as pbar:
    for i, file in enumerate(files):
        data = pd.read_csv(join(input_dir, file),
                            dtype=dtypes)
        data["file"] = file.split(".")[0]

        data.replace("", pd.NA, inplace=True)

        data.to_csv(output_file, 
                        mode='a', 
                        header=(i == 0), # write header only once
                        index=False,
                        compression='gzip')
        del data
        pbar.update(1)





