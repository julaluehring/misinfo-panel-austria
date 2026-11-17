# Task:   Language shares in the panel, overall and by tweet type.
# Input:  data/Austria-Panel-Tweets.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Language-Shares.csv, Austria-Panel-Language-Shares-Type.csv
# Run:    python describe_language.py ../../../data

import polars as pl
from os.path import join
import sys


src = sys.argv[1]
file = "Austria-Panel-Tweets.csv.gz"

# Twitter codes that are not a language:
# und = undetermined, zxx = no linguistic content (e.g., only links/media),
# qme = media only, qht = hashtags only, qam = mentions only,
# qct = cashtags only, art = emojis only
NON_LANG = ["und", "zxx", "qme", "qht", "qam", "qct", "art"]

# keep only authors that we subsetted
authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                  separator=",",
                  quote_char='"')
            .select(pl.col("author_id").cast(pl.Utf8))
            .unique()
        )

tweets = (pl.scan_csv(join(src, file),
                      separator=",",
                      quote_char='"',
                      schema_overrides={"id": pl.Utf8,
                                        "author_id": pl.Utf8,
                                        "lang": pl.Utf8,
                                        "tweet_type": pl.Utf8})
            .select(["id", "author_id", "created_at", "lang", "tweet_type"])
            .with_columns(pl.col("created_at").str.slice(0, 10).alias("day"))
            .join(authors, on="author_id", how="inner")
            .filter(pl.col("day") >= "2019-01-01")
            .filter(pl.col("day") < "2023-04-01")
            .unique(subset="id")
            .with_columns(
                pl.col("lang").fill_null("missing"),
                pl.col("tweet_type").fill_null("original"),
                pl.col("lang").is_in(NON_LANG).alias("no_language")
            )
        )

# 1) language shares across all tweets
lang_all = (tweets
            .group_by("lang")
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / pl.col("n").sum() * 100)
                          .round(2).alias("pct_all"))
            .sort("n", descending=True)
            .collect(streaming=True))

print(f"Total tweets: {lang_all['n'].sum()}")
print(lang_all.head(15))

# 2) German share among tweets with a detected language
detected = lang_all.filter(~pl.col("lang").is_in(NON_LANG + ["missing"]))
n_detected = detected["n"].sum()
n_de = detected.filter(pl.col("lang") == "de")["n"].sum()
n_en = detected.filter(pl.col("lang") == "en")["n"].sum()
print(f"Tweets with detected language: {n_detected} "
      f"({n_detected / lang_all['n'].sum() * 100:.2f}% of all)")
print(f"German among detected: {n_de / n_detected * 100:.2f}%")
print(f"English among detected: {n_en / n_detected * 100:.2f}%")

# 3) by tweet type (retweets were not scored for emotions,
#    see inference/emotions/1-prepare_text.py)
lang_type = (tweets
             .group_by(["tweet_type", "lang"])
             .agg(pl.len().alias("n"))
             .with_columns(
                 (pl.col("n") / pl.col("n").sum().over("tweet_type") * 100)
                 .round(2).alias("pct_within_type"))
             .sort(["tweet_type", "n"], descending=[False, True])
             .collect(streaming=True))

print(lang_type.group_by("tweet_type").agg(pl.col("n").sum()))  # check types
print(lang_type.filter(pl.col("lang").is_in(["de", "en"])))

lang_all.write_csv(join(src, "Austria-Panel-Language-Shares.csv"))
lang_type.write_csv(join(src, "Austria-Panel-Language-Shares-Type.csv"))
print(f"Saved data to {join(src, 'Austria-Panel-Language-Shares.csv')}")
