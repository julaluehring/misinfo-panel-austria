# ===== SCRIPT OVERVIEW =====
# Task:   Match extracted tweet domains against the NewsGuard domain list using
#         fuzzy matching (rapidfuzz, similarity threshold 97).
# Input:  <DATA_DIR>/urls/  (concatenated URL files from step 3)
#         NewsGuard domain list (expected in <DATA_DIR>/newsguard/)
# Output: <DATA_DIR>/urls/matched_domains_all.csv.gz
# Usage:  python 7-match_domains.py <DATA_DIR>

import pandas as pd
from os.path import join
import numpy as np
import tldextract
from rapidfuzz import fuzz, process
import sys

src = sys.argv[1]
n_workers = -10
similarity_threshold = 97

# step 1: create a superset of urls
unraveled_urls = pd.read_csv(join(src,"urls/"
                    "unraveled_urls_all.csv.gz"), 
                    compression="gzip")
unshortened_urls = pd.read_csv(join(src, "urls",
                    "unshortened_urls_r2.csv.gz"),
                    compression="gzip")


unique_urls = pd.concat(
    [unraveled_urls,
     unshortened_urls],
    ignore_index=True)\
        .drop_duplicates(subset=["url"])\
        .dropna(subset=["domain"])

# step 2: match identical domains with newsguard
ng_domains = pd.read_csv(
      join(src, "newsguard/",
        "domains_unique.csv"),
      usecols=["Domain"])\
        .drop_duplicates(subset=["Domain"])\
        .rename(columns={"Domain": "ng_domain"})

matched_domains = unique_urls\
    .merge(
        ng_domains,
        how="inner",
        left_on="domain",
        right_on="ng_domain"
        )

# step 3: extract unmatched domains
unmatched_domains = unique_urls[
    ~unique_urls["domain"].isin(
        matched_domains["ng_domain"])
].copy()

# step 4: extract registered domain (host) from unmatched URLs
def get_registered_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

unmatched_domains["host"] = \
    unmatched_domains["domain"]\
        .astype(str)\
        .apply(get_registered_domain)

# step 5: match host domains with newsguard and concatenate
matched_hosts = unmatched_domains.merge(
    ng_domains,
    how="inner",
    left_on="host",
    right_on="ng_domain"
)

matched_domains = pd.concat(
    [matched_domains, matched_hosts],
    ignore_index=True
)

# step 6: extract unmatched domains and do fuzzymatching
unmatched_domains = unmatched_domains[
    ~unmatched_domains["host"].isin(
        matched_domains["ng_domain"])
].copy()

unmatched_domains_unique = unmatched_domains\
    .drop_duplicates(subset=["host"]
).copy()

# multi-process the fuzzy matching with cdist
queries = unmatched_domains_unique["host"]\
    .dropna()\
    .astype(str).tolist()

choices = ng_domains["ng_domain"]\
    .dropna()\
    .astype(str).unique().tolist()

similarity_matrix = process.cdist(queries, 
                                  choices, 
                                  scorer=fuzz.ratio, 
                                  workers=n_workers)

# for each query, find the index of the best match in choices
best_match_indices = np.argmax(similarity_matrix, axis=1)
best_scores = np.max(similarity_matrix, axis=1)

# get the best matching domains
best_matches = [choices[idx] for idx in best_match_indices]

results_df = pd.DataFrame({
    "host": queries,
    "ng_domain": best_matches,
    "similarity_score": best_scores
})

# keep only the ones with a very high similarity score
filtered_results = results_df[results_df["similarity_score"] > similarity_threshold]

# step 7: merge with matched domains and save matched domains dataset 
fuzzy_matched_domains = unmatched_domains_unique.merge(
    filtered_results[["host", "ng_domain"]],
    how="inner",
    left_on=["host"],
    right_on=["host"]
)

# concatenate with matched domains
matched_domains = pd.concat(
    [matched_domains, fuzzy_matched_domains],
    ignore_index=True
).drop(columns=["host"])

perc_matched = len(matched_domains) / len(unique_urls) * 100
print(f'Matched {len(matched_domains)} URLs out of {len(unique_urls)}, which is {perc_matched:.2f}% of the total URLs.') #8.43

# step 8: save urls with ng domain
matched_domains.to_csv(
    join(src, "urls", 
         "matched_domains_all.csv.gz"),
    index=False,
    compression="gzip"
)

print(f"Matched URLs saved at {join(src, 'urls', 'matched_domains_all.csv.gz')}!")