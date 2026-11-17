# Task:   Draw the stratified German and English validation sample for manual annotation.
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/emotions/emotion_inference.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/emotions/validation/validation_v2_annotate.csv (tweet text, not released)
#         data/emotions/validation/validation_v2_key.csv
# Run:    python 4-sample_validation_v2.py ../../../data [n_per_stratum_hi] [n_low]

import polars as pl
from os.path import join, exists
import os
import sys

src = sys.argv[1]
N_HI = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_LOW = int(sys.argv[3]) if len(sys.argv) > 3 else 200
LANGS = ["de", "en"]
EMOTIONS = ["anger", "fear", "disgust"]
SEED = 636
val_dir = join(src, "emotions", "validation")
os.makedirs(val_dir, exist_ok=True)
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"

# tweets of the panel in the observation period, German and English
tweets = (pl.scan_csv(join(src, file), separator=",", quote_char='"',
                      schema_overrides={"lang": pl.Utf8})
            .select(["id", "author_id", "created_at", "lang"])
            .with_columns([pl.col("id").cast(pl.Utf8),
                           pl.col("author_id").cast(pl.Utf8),
                           pl.col("created_at").str.slice(0, 10).alias("day")])
            .filter((pl.col("day") >= "2019-01-01") & (pl.col("day") < "2023-04-01"))
            .filter(pl.col("lang").is_in(LANGS)))
authors = (pl.scan_csv(join(src, "authors_filtered.csv"))
             .select(pl.col("author_id").cast(pl.Utf8)).unique())
emotions = (pl.scan_csv(join(src, "emotions", "emotion_inference.csv.gz"), separator=",")
              .select(["id", "text_cleaned"] + [f"{e}_v2" for e in EMOTIONS])
              .with_columns([pl.col("id").cast(pl.Utf8)] +
                            [pl.col(f"{e}_v2").cast(pl.Float64) for e in EMOTIONS]))

# exclude tweets from the first validation
done = set()
for f in ["sampled_tweets_manual_jl.csv", "sampled_tweets_manual.csv",
          "sampled_tweets_manual_r2.csv", "sampled_tweets_inference.csv"]:
    f = join(val_dir, f)
    if exists(f):
        sep = ";" if f.endswith("_jl.csv") else ","
        done |= set(pl.read_csv(f, separator=sep, columns=["id"],
                                schema_overrides={"id": pl.Utf8})["id"].to_list())
print(f"Excluding {len(done):,} tweets from the first validation")

pop = (tweets.join(authors, on="author_id", how="inner")
             .join(emotions, on="id", how="inner")
             .filter(pl.all_horizontal([pl.col(f"{e}_v2").is_not_null() for e in EMOTIONS]))
             .filter(~pl.col("id").is_in(list(done)))
             .with_columns(
                 pl.when(pl.col("fear_v2") >= 0.5).then(pl.lit("fear_hi"))
                   .when(pl.col("disgust_v2") >= 0.5).then(pl.lit("disgust_hi"))
                   .when(pl.col("anger_v2") >= 0.5).then(pl.lit("anger_hi"))
                   .otherwise(pl.lit("low")).alias("stratum")))

# population size per language x stratum
sizes = pop.group_by(["lang", "stratum"]).agg(pl.len().alias("N_stratum")).collect()
print(sizes.sort(["lang", "stratum"]))

samples = []
for lang in LANGS:
    for stratum in ["fear_hi", "disgust_hi", "anger_hi", "low"]:
        n = N_LOW if stratum == "low" else N_HI
        N = sizes.filter((pl.col("lang") == lang) & (pl.col("stratum") == stratum))["N_stratum"]
        N = int(N[0]) if len(N) else 0
        q = pop.filter((pl.col("lang") == lang) & (pl.col("stratum") == stratum))
        # thin very large strata by a hash of the id before collecting (memory)
        if N > 200 * n:
            k = N // (100 * n)
            q = q.filter(pl.col("id").hash(SEED) % k == 0)
        s = q.collect()
        s = s.sample(n=min(n, s.height), seed=SEED)
        samples.append(s.with_columns([pl.lit(N).alias("N_stratum"),
                                       pl.lit(s.height).alias("n_stratum")]))
        print(f"{lang} {stratum}: sampled {s.height} of {N:,}")

key = (pl.concat(samples)
         .with_columns((pl.col("N_stratum") / pl.col("n_stratum")).alias("weight"))
         .sample(fraction=1.0, shuffle=True, seed=SEED))

key.select(["id", "lang", "author_id", "day", "stratum", "N_stratum", "n_stratum",
            "weight"] + [f"{e}_v2" for e in EMOTIONS]) \
   .write_csv(join(val_dir, "validation_v2_key.csv"))

# blind annotation file: no scores, no strata; one row per tweet, all three emotions
key.select(["id", "lang", "text_cleaned"]) \
   .with_columns([pl.lit(None, dtype=pl.Int64).alias(f"{e}_manual") for e in EMOTIONS]) \
   .write_csv(join(val_dir, "validation_v2_annotate.csv"))
print(f"Saved {key.height} tweets to validation_v2_annotate.csv (blind) and validation_v2_key.csv")
