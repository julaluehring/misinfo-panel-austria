# Task:   Concatenate the per-file tweet CSVs into one file.
# Input:  data/CSVtweets/* (not released)
#         data/dtypes.pkl
# Output: data/Austria-Panel-Tweets.csv.gz (not released)
# Run:    python 3-concat_tweet_data.py ../../data

import os
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
import sys
import pickle as pl

src = sys.argv[1] # <DATA_DIR>
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





