# Task:   Aggregate tweet-level emotion and NewsGuard scores to daily counts (emotion present if
#         score >= 0.65; also 0.5 and Q3 for the SI).
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/emotions/emotion_inference.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Daily-<lang>.csv
# Run:    python 1-aggregate_data.py ../../data [lang]      # lang: de_en (default), de, all

import polars as pl
from os.path import join
import sys

src = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "de_en"
LANG_SETS = {"de_en": ["de", "en"], "de": ["de"], "all": None}
assert LANG in LANG_SETS, f"lang must be one of {list(LANG_SETS)}"
EMOTIONS = ["anger", "fear", "disgust"]
THRESHOLD = 0.5       # SI robustness
THRESHOLD_ALT = 0.65  # main specification (classifier developers' threshold)
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"
print(f"Language set: {LANG}")

df = (pl.scan_csv(join(src, file),
                  separator=",",
                  quote_char='"',
                  schema_overrides={"lang": pl.Utf8})
        .select([
            "id",
            "author_id",
            "created_at",
            "lang",
            "ng_domain",
            "Orientation",
            "Score"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("author_id").cast(pl.Utf8),
            pl.col("created_at").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ"),
            pl.col("Score").cast(pl.Float64)])
        .with_columns([
            pl.col("created_at").dt.strftime("%Y-%m-%d").alias("day")
        ])
    )

# keep only authors that we subsetted
authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                  separator=",",
                  quote_char='"')
            .select(pl.col("author_id").cast(pl.Utf8))
            .unique()
        )

# keep only tweets from filtered authors within the observation period
df_filtered = (df
                .join(authors, on="author_id", how="inner")
                .filter(pl.col("day") >= "2019-01-01")
                .filter(pl.col("day") < "2023-04-01")
)

# add the emotion scores
emotions = (
        pl.scan_csv(join(src, "emotions", "emotion_inference.csv.gz"), separator=",")
            .select(["id"] + [f"{e}_v2" for e in EMOTIONS])
            .with_columns([pl.col("id").cast(pl.Utf8)] +
                          [pl.col(f"{e}_v2").cast(pl.Float64) for e in EMOTIONS])
        )

df_emotions = df_filtered.join(emotions, on="id", how="left")  # keep all tweets

# language filter: set emotion scores of other languages to missing
if LANG_SETS[LANG] is not None:
    in_lang = pl.col("lang").is_in(LANG_SETS[LANG]).fill_null(False)
    df_emotions = df_emotions.with_columns([
        pl.when(in_lang).then(pl.col(f"{e}_v2")).otherwise(None).alias(f"{e}_v2")
        for e in EMOTIONS])

# third quartiles over classified tweets in the selected languages (SI only)
q3 = (df_emotions
      .select([pl.col(f"{e}_v2").quantile(0.75).alias(e) for e in EMOTIONS])
      .collect()
      .row(0, named=True))
print("Q3 thresholds: " + ", ".join(f"{e} = {q3[e]:.4f}" for e in EMOTIONS))


def count_at(col, cutoff, name):
    return (pl.col(col) >= cutoff).fill_null(False).cast(pl.Int64).sum().alias(name)


# aggregate the data per day
daily_summary = (df_emotions
    .group_by("day")
    .agg([
        pl.len().alias("tweet_count"),
        # classified tweets in the selected languages (denominator of emotion shares)
        pl.col("anger_v2").is_not_null().sum().alias("n_scored"),

        pl.col("ng_domain").is_not_null().sum().alias("count_ng_domains"),
        pl.col("Score").is_not_null().sum().alias("count_ng_scored"),  # tweets with a NewsGuard score

        pl.when(pl.col("Score") <= 39).then(1).otherwise(0).sum()
            .alias("count_max_caution"),  # 0–39
        pl.when((pl.col("Score") >= 40) & (pl.col("Score") <= 59)).then(1).otherwise(0).sum()
            .alias("count_caution"),  # 40–59
        pl.when((pl.col("Score") >= 60) & (pl.col("Score") <= 74)).then(1).otherwise(0).sum()
            .alias("count_exceptions"),  # 60–74
        pl.when((pl.col("Score") >= 75) & (pl.col("Score") <= 99)).then(1).otherwise(0).sum()
            .alias("count_generally_credible"),  # 75–99
        pl.when(pl.col("Score") == 100).then(1).otherwise(0).sum()
            .alias("count_high_credibility"),  # 100 only
        pl.col("Score").mean().alias("mean_score"),

        pl.when(pl.col("Orientation") == "Right").then(1).otherwise(0).sum().alias("count_right"),
        pl.when(pl.col("Orientation") == "Neutral").then(1).otherwise(0).sum().alias("count_neutral"),
        pl.when(pl.col("Orientation") == "Left").then(1).otherwise(0).sum().alias("count_left"),

        pl.col("author_id").n_unique().alias("n_authors")
    ]
    # emotions: 0.5 (SI), 0.65 (main), third quartile (SI), mean score (SI)
    + [count_at(f"{e}_v2", THRESHOLD, f"count_{e}_05") for e in EMOTIONS]
    + [count_at(f"{e}_v2", THRESHOLD_ALT, f"count_{e}_065") for e in EMOTIONS]
    + [count_at(f"{e}_v2", q3[e], f"count_{e}_q3") for e in EMOTIONS]
    + [pl.col(f"{e}_v2").mean().alias(f"mean_{e}") for e in EMOTIONS])
    .sort("day")
    .collect()
)

print(f"{daily_summary.height} days; classified tweets: {daily_summary['n_scored'].sum():,} "
      f"of {daily_summary['tweet_count'].sum():,} tweets")
for e in EMOTIONS:
    share = daily_summary[f"count_{e}_05"].sum() / daily_summary["n_scored"].sum() * 100
    share_alt = daily_summary[f"count_{e}_065"].sum() / daily_summary["n_scored"].sum() * 100
    print(f"  {e}: {share:.2f}% of classified tweets with score >= {THRESHOLD}, "
          f"{share_alt:.2f}% with score >= {THRESHOLD_ALT}")

out = join(src, f"Austria-Panel-Daily-{LANG}.csv")
daily_summary.write_csv(out, separator=",", quote_char='"')
print(f"Saved {out}")
