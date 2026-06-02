# ===== SCRIPT OVERVIEW =====
# Task:   Extract and normalise domains from tweet expanded_urls.
#         Separates shortened URLs (e.g. bit.ly) from directly resolved domains.
# Input:  <DATA_DIR>/Austria-Panel-Tweets.csv.gz  (column: expanded_urls)
#         ./url-shorteners/list.txt         (list of known URL shortener domains)
# Output: <DATA_DIR>/urls/unshortened_urls_r2.csv.gz
#         <DATA_DIR>/urls/shortened_urls_r2.csv.gz
# Usage:  python 1-extract_urls.py <DATA_DIR>

import numpy as np
from os.path import join
import sys
import pandas as pd
from tqdm import tqdm

src = sys.argv[1] 
dst = join(src, "urls")
data = 'Austria-Panel-Tweets.csv.gz' 

CHUNK_SIZE = 30_000_000
MAX_ROWS = 206_178_664
N_CHUNKS = MAX_ROWS // CHUNK_SIZE

# read in URL shorteners
with open("./url-shorteners/list.txt", "r") as f:
    url_shorteners = f.readlines()

url_shorteners = [x.strip() for x in url_shorteners]

def extract_domain(url):
    if url != url:
        return np.nan
    # trailing "/" and spaces
    url = url.strip('/').strip()
    # transform all domains to lowercase
    url = url.lower()
    # remove any white spaces
    url = url.replace(' ', '')
    # if present: remove the protocol
    if url.startswith(("http", "https")):
        try:
            url = url.split('//')[1]
        except IndexError:
            print(f"found malformed URL {url}")
            return np.nan
    # remove "www." 
    url = url.replace('www.', '')
    url = url.split("/")[0]
    return url


first_header = True
# read in chunks of df with pandas
for chunk in tqdm(
    pd.read_csv(join(src, data), 
                      compression='gzip', 
                      chunksize=CHUNK_SIZE,
                      usecols=["id", "expanded_urls"],
                      dtype={"id": str,
                             "expanded_urls": object}),
    total=N_CHUNKS,
    desc="Extracting domains",
    unit="chunk"):


    # drop NaNs
    chunk = chunk[chunk["expanded_urls"].notna()]



    URLs = []
    # extract ID and expanded URL
    chunk.loc[:, "expanded_urls"] = chunk["expanded_urls"].str.split(" ")
    for url_lst in chunk["expanded_urls"]:
        URLs.extend(url_lst)

    # create dataframe
    URLs = pd.DataFrame({"url":list(set(URLs))})
    URLs["domain"] = URLs["url"].apply(extract_domain)

    # save shortened URLs to an extra csv
    shortened_urls = URLs[URLs["domain"].isin(url_shorteners)]
    unshortened_urls = URLs[~URLs["domain"].isin(url_shorteners)]   

    unshortened_urls.\
        to_csv(join(dst, "unshortened_urls_r2.csv.gz"),
                mode='a', 
                index=False, 
                header=first_header,
                compression='gzip'
    )

    shortened_urls\
        .to_csv(join(dst, "shortened_urls_r2.csv.gz"),
                mode='a', 
                index=False, 
                header=first_header,
                compression='gzip'
    )

    first_header = False
    del chunk, URLs, shortened_urls, unshortened_urls

print("Finished extracting domains.")