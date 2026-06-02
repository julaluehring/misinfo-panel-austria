# ===== SCRIPT OVERVIEW =====
# Task:   Concatenate all per-file user CSVs into a single compressed file. Deduplicates by author_id (keeps last occurrence).
# Input:  UserCSVfiles/*.csv
#         dtypes.pkl  
# Output: Austria-Panel-Users.csv.gz
# Usage:  python 5-concat_user_data.py <DATA_DIR>

import os
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
import sys
import pickle as pl

src = sys.argv[1] # /misinfo-panel-austria/data
input_dir = join(src, "UserCSVfiles")
output_file = join(src, "Austria-Panel-Users.csv.gz")

# extract a list of files in the directory
files = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f))]
# files = files[:3] # test strategy

# define data types
dtypes = pl.load(open(join(src, "dtypes.pkl"), "rb"))
float_columns = [
    "followers_count",
    "following_count",
    "tweet_count"
]

# loop through the files and append to a new csv-file
with tqdm(total=len(files), desc="Processing files", unit="file") as pbar:
    for i, file in enumerate(files):
        data = pd.read_csv(join(input_dir, file),
                            dtype=dtypes)
        data["file"] = file.split(".")[0]

        data.replace("", pd.NA, inplace=True)

        for col in float_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], 
                                errors='coerce')
        data.drop_duplicates(subset=["author_id"],
            inplace=True,keep="last")
        data.to_csv(output_file, 
                        mode='a', 
                        header=(i == 0), # write header only once
                        index=False,
                        compression='gzip')
        del data
        pbar.update(1)