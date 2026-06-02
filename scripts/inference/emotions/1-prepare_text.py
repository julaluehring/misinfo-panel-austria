# ===== SCRIPT OVERVIEW =====
# Task:   Clean raw tweet text for emotion inference: remove emojis, URLs, @-mentions,
#         line breaks; exclude retweets (RT prefix), non-strings, and empty strings.
# Input:  <DATA_DIR>/Austria-Panel-Tweets.csv.gz  (columns used: id, text)
# Output: <DATA_DIR>/emotions/Austria-Panel-Text.csv.gz  (columns: id, text_cleaned)
# Usage:  python 1-prepare_text.py <DATA_DIR>

import os
import pandas as pd
from os.path import join
import re
import sys
from tqdm import tqdm
import gzip

# suppress copy warning
pd.options.mode.chained_assignment = None

src = sys.argv[1] # /misinfo-panel-austria/data/

file = join(src, "Austria-Panel-Tweets.csv.gz")
out_file = join(src, "emotions", "Austria-Panel-Text.csv.gz")
os.makedirs(join(src, "emotions"), exist_ok=True)
CHUNK_SIZE = 10000000
TOTAL_ROWS = 206178664
TOTAL_CHUNKS = TOTAL_ROWS // CHUNK_SIZE


emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F700-\U0001F77F"  # alchemical symbols
                        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                        u"\U00002702-\U000027B0"  # Dingbats
                        u"\U000024C2-\U0001F251" 
                        "]+", flags=re.UNICODE)

def clean_text(text):
    # convert non-strings (float nans) to empty string
    if not isinstance(text, str):  
        # print(f"Unexpected non-string value encountered: {repr(text)}")
        return ""
    text = emoji_pattern.sub(r'', text) # remove emojis
    text = re.sub("https.*", "", text)  # remove links
    text = re.sub("@\w+", "", text) # remove @-handles
    # text = re.sub("#", " ", text) # remove hashtags
    text = text.replace('\n', ' ') # remove line breaks
    text = text.replace('\\', ' ') # remove backslash
    return text.strip()

with gzip.open(out_file, "wt", encoding="utf-8") as f_out:
    first_chunk = True

    for chunk in tqdm(pd.read_csv(file, 
                 compression="gzip",
                 usecols=["id", "text"], # add "lang" if needed
                 nrows=TOTAL_ROWS,
                 dtype={"id": str, "text":str}, # "lang": str
                 chunksize=CHUNK_SIZE), 
                 desc="Processing chunks",
                 unit="chunk",
                 total=TOTAL_CHUNKS):
        
        # chunk_de = chunk.loc[chunk["lang"] == "de"]

        chunk.loc[:, "text_cleaned"] = chunk["text"]\
            .apply(clean_text)

        chunk = chunk.loc[chunk["text_cleaned"].notna()]
        chunk = chunk.loc[chunk["text_cleaned"] != ""]
        chunk = chunk.loc[~chunk["text"].str.startswith("RT")]

        chunk[['id', 'text_cleaned']]\
                                .to_csv(f_out,
                                        index=False,
                                        header=first_chunk)
        
        first_chunk = False