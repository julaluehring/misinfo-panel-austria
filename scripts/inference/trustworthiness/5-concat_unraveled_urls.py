# ===== SCRIPT OVERVIEW =====
# Task:   Concatenate per-batch unraveled URL files into a single file.
# Input:  <SRC_DIR>/unraveled_*.csv.gz  (output of 2_unravel_urls.py)
# Output: <DST>  (path provided as second argument)
# Usage:  python 5-concat_unraveled_urls.py <SRC_DIR> <DST>

import pandas as pd
import numpy as np
from os import listdir
from os.path import join
import sys

src = sys.argv[1] 
files = listdir(src)
dst = sys.argv[2] 

# concatenate all files
# and create dataframe
unraveled_urls = pd.DataFrame()
for i,f in enumerate(files):
    if i%1000 == 0:
        print(f"{i}/{len(files)}")
    tmp = pd.read_csv(join(src, f), compression="gzip")
    unraveled_urls = pd.concat([unraveled_urls, tmp])
unraveled_urls = unraveled_urls.reset_index(drop=True)

# print timeouts for documentation
timeouts = len(unraveled_urls) - len(unraveled_urls["status_code"].dropna())
print("{} timeouts ({:1.2f}%)".format(\
        timeouts,
        (timeouts / len(unraveled_urls["status_code"].dropna()) * 100)))

# extract the unraveled URLs
def extract_host(unraveled_url):
    if unraveled_url == unraveled_url and unraveled_url.startswith("Cannot"):
        host = unraveled_url.split(" ")[4].split(":")[0]
        return host
    else:
        return unraveled_url

unraveled_urls["unraveled_url"] = unraveled_urls["unraveled_url"]\
                                    .apply(extract_host)

# extract the domains
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
            # print(f"found malformed URL {url}")
            return np.nan
    # remove "www." 
    url = url.replace('www.', '')
    url = url.split("/")[0]
    return url

unraveled_urls["domain"] = unraveled_urls["unraveled_url"].apply(extract_domain)
# print the missing domains
missing_domains = unraveled_urls["domain"].isna().sum()
print("{} missing domains ({:1.2f}%)".format(\
        missing_domains,
        (missing_domains / len(unraveled_urls))
        * 100))

# save the unraveled URLs
unraveled_urls.to_csv(join(dst, "unraveled_domains_reduc.csv.gz"), 
                        index=False,
                        compression="gzip")