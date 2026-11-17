# Task:   Timescales of the emotion features: variance within days, weeks and years, and
#         instability and inertia computed with daily, weekly and monthly windows.
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/emotions/emotion_inference.csv.gz (not released)
#         data/authors_filtered.csv
#         data/Austria-Panel-Users-Features-Tweet-<lang>.csv
# Output: data/Austria-Panel-Variance-Decomposition-<lang>.csv
#         data/Austria-Panel-Users-Features-Windows-<lang>.csv (not released)
# Run:    python describe_feature_timescales.py ../../../data [lang] [n_test_users]   # lang: de_en (default), de, all

import polars as pl
import numpy as np
from os.path import join
import sys


src = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "de_en"
LANG_SETS = {"de_en": ["de", "en"], "de": ["de"], "all": None}
assert LANG in LANG_SETS, f"lang must be one of {list(LANG_SETS)}"
n_test_users = int(sys.argv[3]) if len(sys.argv) > 3 else None
EMOTIONS = ["anger", "fear", "disgust"]
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"

# minimum tweets per window (week as in the paper)
MIN_TWEETS = {"day": 5, "week": 10, "month": 10}


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan


''' DATA (as in 4-extract_features.py) '''
df = (pl.scan_csv(join(src, file), separator=",", quote_char='"',
                  schema_overrides={"lang": pl.Utf8})
        .select(["id", "author_id", "created_at", "lang"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("author_id").cast(pl.Utf8),
            pl.col("created_at").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ")])
        .with_columns(pl.col("created_at").dt.date().alias("day")))

authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                       separator=",", quote_char='"')
             .select(pl.col("author_id").cast(pl.Utf8))
             .unique())
if n_test_users is not None:
    print(f"TEST MODE: sampling {n_test_users} users")
    authors = authors.collect().sample(n=n_test_users, seed=63).lazy()

emotions = (pl.scan_csv(join(src, "emotions", "emotion_inference.csv.gz"),
                        separator=",")
              .select(["id"] + [f"{e}_v2" for e in EMOTIONS])
              .with_columns([pl.col("id").cast(pl.Utf8)] +
                            [pl.col(f"{e}_v2").cast(pl.Float64) for e in EMOTIONS]))

tweets = (df
          .join(authors, on="author_id", how="inner")
          .filter(pl.col("day") >= pl.date(2019, 1, 1))
          .filter(pl.col("day") < pl.date(2023, 4, 1))
          .join(emotions, on="id", how="left"))
# language filter: set emotion scores of other languages to missing
if LANG_SETS[LANG] is not None:
    in_lang = pl.col("lang").is_in(LANG_SETS[LANG]).fill_null(False)
    tweets = tweets.with_columns([
        pl.when(in_lang).then(pl.col(f"{e}_v2")).otherwise(None).alias(f"{e}_v2")
        for e in EMOTIONS])
tweets = (tweets
          .with_columns([
              pl.col("day").dt.strftime("%G").alias("iso_year"),
              pl.col("day").dt.strftime("%G-%V").alias("week"),
              pl.col("day").dt.strftime("%Y-%m").alias("month"),
              pl.col("day").dt.strftime("%Y-%m-%d").alias("day_str")])
          .sort(["author_id", "created_at"])
          .collect())
print(f"Loaded {tweets.height:,} tweets "
      f"({tweets['anger_v2'].is_not_null().sum():,} classified in {LANG})")
# dynamics use consecutive classified tweets only (as in 4-extract_features.py)
scored = tweets.filter(pl.all_horizontal([pl.col(f"{e}_v2").is_not_null() for e in EMOTIONS]))


''' PART 1: VARIANCE DECOMPOSITION (classified tweets only) '''
# nested levels: user > ISO year > ISO week > day > tweet
# SS_total = SS_user + SS_year|user + SS_week|year + SS_day|week + SS_within_day
LEVELS = ["author_id", "iso_year", "week", "day_str"]
rows = []
for emo in EMOTIONS:
    v = f"{emo}_v2"
    d = (tweets.filter(pl.col(v).is_not_null())
         .group_by(LEVELS)
         .agg([pl.len().alias("n"),
               pl.col(v).sum().alias("s"),
               (pl.col(v) ** 2).sum().alias("ss")]))
    n_all, s_all, ss_all = d["n"].sum(), d["s"].sum(), d["ss"].sum()
    grand = s_all / n_all
    ss_total = ss_all - n_all * grand ** 2
    # within days
    ss_within_day = (d["ss"] - d["s"] ** 2 / d["n"]).sum()
    # between each level and its parent: sum over groups n_g * (mean_g - mean_parent)^2
    comp = {"within days": ss_within_day}
    names = {1: "between users", 2: "between years (within users)",
             3: "between weeks (within years)", 4: "between days (within weeks)"}
    for k in range(4, 0, -1):
        child = (d.group_by(LEVELS[:k])
                 .agg([pl.col("n").sum(), pl.col("s").sum()]))
        if k > 1:
            parent = (d.group_by(LEVELS[:k - 1])
                      .agg([pl.col("n").sum().alias("n_p"),
                            pl.col("s").sum().alias("s_p")]))
            child = child.join(parent, on=LEVELS[:k - 1])
            between = (child["n"] * (child["s"] / child["n"]
                                     - child["s_p"] / child["n_p"]) ** 2).sum()
        else:
            between = (child["n"] * (child["s"] / child["n"] - grand) ** 2).sum()
        comp[names[k]] = between
    within_user = ss_total - comp["between users"]
    for name, val in comp.items():
        rows.append({
            "emotion": emo,
            "component": name,
            "pct_of_total": 100 * val / ss_total,
            "pct_of_within_user": (100 * val / within_user
                                   if name != "between users" else np.nan)})
    print(f"{emo}: components sum to {100 * sum(comp.values()) / ss_total:.2f}% "
          f"of total (should be 100)")

decomp = pl.DataFrame(rows)
print("\n=== Variance decomposition of tweet-level scores ===")
with pl.Config(tbl_rows=20, float_precision=2):
    print(decomp)
decomp.write_csv(join(src, f"Austria-Panel-Variance-Decomposition-{LANG}.csv"))


''' PART 2: WINDOW SENSITIVITY OF INSTABILITY AND INERTIA '''
def window_features(tw: pl.DataFrame, win: str, min_tweets: int) -> pl.DataFrame:
    # windows with enough classified tweets (as in 4-extract_features.py)
    keep = (tw.group_by(["author_id", win]).agg(pl.len().alias("n"))
              .filter(pl.col("n") >= min_tweets)
              .select(["author_id", win]))
    t = (tw.join(keep, on=["author_id", win], how="inner")
           .sort(["author_id", win, "created_at"])
           .with_columns([
               *[pl.col(f"{e}_v2").shift(1).over(["author_id", win]).alias(f"{e}_prev")
                 for e in EMOTIONS],
               pl.col("created_at").shift(1).over(["author_id", win]).alias("t_prev")])
           .with_columns([
               ((pl.col("created_at") - pl.col("t_prev")).dt.total_seconds()
                / (24 * 3600)).alias("delta_days"),
               # np.timedelta64[h] in 4-extract_features.py truncates to full hours
               ((pl.col("created_at") - pl.col("t_prev")).dt.total_seconds()
                // 3600).cast(pl.Float64).alias("delta_hours")]))

    # instability: MSSD within windows, then mean across windows per user
    mssd = (t.filter((pl.col("delta_days") > 0) &
                     pl.all_horizontal([pl.col(f"{e}_prev").is_not_null() for e in EMOTIONS]))
             .with_columns([((pl.col(f"{e}_v2") - pl.col(f"{e}_prev")) ** 2).alias(f"{e}_sq")
                            for e in EMOTIONS])
             .group_by(["author_id", win])
             .agg([pl.col(f"{e}_sq").mean() for e in EMOTIONS])
             .group_by("author_id")
             .agg([pl.col(f"{e}_sq").mean().alias(f"{e}_instability_{win}") for e in EMOTIONS]))

    # inertia: phi_hour = exp(-|lambda|), lambda = sum(X*y)/sum(X^2),
    # X = -delta_hours, y = log|x_t1 / x_t|, per window, then mean per user
    out = mssd
    for emo in EMOTIONS:
        x0, x1 = pl.col(f"{emo}_prev"), pl.col(f"{emo}_v2")
        pairs = (t.filter((pl.col("delta_hours") > 0) &
                          x0.is_not_null() & x1.is_not_null() &
                          (x0 != 0) & (x1 != 0))
                   .with_columns([
                       (-pl.col("delta_hours")).alias("X"),
                       (x1 / x0).abs().log().alias("y")]))
        inert = (pairs.group_by(["author_id", win])
                   .agg([pl.len().alias("n_pairs"),
                         x0.std(ddof=0).alias("sd0"),
                         x1.std(ddof=0).alias("sd1"),
                         (pl.col("X") * pl.col("y")).sum().alias("xy"),
                         (pl.col("X") ** 2).sum().alias("xx")])
                   .filter((pl.col("n_pairs") >= 2) &
                           (pl.col("sd0") > 0) & (pl.col("sd1") > 0))
                   .with_columns((-(pl.col("xy") / pl.col("xx")).abs()).exp()
                                 .alias("phi"))
                   .group_by("author_id")
                   .agg(pl.col("phi").drop_nans().mean()
                        .alias(f"{emo}_inertia_{win}")))
        out = out.join(inert, on="author_id", how="full", coalesce=True)
    return out


features = None
for win, k in MIN_TWEETS.items():
    print(f"computing {win} windows (min. {k} tweets)...")
    f = window_features(scored, win, k)
    features = f if features is None else features.join(
        f, on="author_id", how="full", coalesce=True)
features.write_csv(join(src, f"Austria-Panel-Users-Features-Windows-{LANG}.csv"))

# sanity check: weekly version vs. features used in the paper
orig = pl.read_csv(join(src, f"Austria-Panel-Users-Features-Tweet-{LANG}.csv"),
                   schema_overrides={"author_id": pl.Utf8})
chk = orig.join(features, on="author_id", how="inner")
print("\n=== Sanity check: weekly recomputation vs. paper features (should be r = 1) ===")
for feat in [f"{e}_{k}" for k in ["instability", "inertia"] for e in EMOTIONS]:
    print(f"{feat}: r = {corr(chk[feat], chk[feat + '_week']):.4f}, "
          f"max |diff| = {(chk[feat] - chk[feat + '_week']).abs().max()}")

print("\n=== Coverage: users with a feature per window ===")
for win in MIN_TWEETS:
    print(f"{win}: instability n = {features[f'anger_instability_{win}'].is_not_null().sum():,}, "
          f"inertia n = {features[f'anger_inertia_{win}'].is_not_null().sum():,}")

print("\n=== Correlation of features across windows (users with all three) ===")
for feat in [f"{e}_{k}" for k in ["instability", "inertia"] for e in EMOTIONS]:
    sub = features.drop_nulls([f"{feat}_day", f"{feat}_week", f"{feat}_month"])
    print(f"{feat} (n = {sub.height:,}): "
          f"day-week r = {corr(sub[feat + '_day'], sub[feat + '_week']):.3f}, "
          f"week-month r = {corr(sub[feat + '_week'], sub[feat + '_month']):.3f}, "
          f"day-month r = {corr(sub[feat + '_day'], sub[feat + '_month']):.3f}")

print("\nDone. Files saved in", src)
