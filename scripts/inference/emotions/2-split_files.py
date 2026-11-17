# Task:   Split the cleaned text into 30M-row chunks for GPU inference.
# Input:  data/emotions/Austria-Panel-Text.csv.gz (not released)
# Output: data/emotions/split_files/*.csv.gz
# Run:    python 2-split_files.py ../../../data

import pandas as pd
import os
import sys

src = sys.argv[1]
file = os.path.join(src, "emotions", "Austria-Panel-Text.csv.gz")
out = os.path.join(src, "emotions", "split_files")
os.makedirs(out, exist_ok=True)
CHUNKSIZE = 30_000_000  # 30M rows per file

for i, chunk in enumerate(pd.read_csv(file, 
                                      compression="gzip", 
                                      chunksize=CHUNKSIZE)):

    output_file = os.path.join(out, f"Austria-Panel-Text_{i+1}.csv.gz") 
    chunk.to_csv(output_file, index=False, compression="gzip")  
    print(f"Saved {output_file} with {len(chunk)} rows")

print("Dataset split into files!")
