# Task:   For each news-sharing event, summarise the user's tweets in the hours before and after.
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/emotions/emotion_inference.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Windows-<hours>h-10min-<lang>.csv.gz
# Run:    python 7-aggregate_windows.py ../../data <hours> [lang] [n_test_users]   # hours: 1 or 24; lang: de_en (default), de, all
import polars as pl
import numpy as np
import pandas as pd
import gzip
import shutil
import os
from os.path import join, exists
import sys
from tqdm import tqdm

src = sys.argv[1]
WINDOW_HOURS = int(sys.argv[2])
LANG = sys.argv[3] if len(sys.argv) > 3 else "de_en"
LANG_SETS = {"de_en": ["de", "en"], "de": ["de"], "all": None}
assert LANG in LANG_SETS, f"lang must be one of {list(LANG_SETS)}"
output_file = join(src, f"Austria-Panel-Windows-{WINDOW_HOURS}h-10min-{LANG}.csv")
n_users = int(sys.argv[4]) if len(sys.argv) > 4 else None  # test mode
EMOTIONS = ["anger", "fear", "disgust"]
MIN_TWEETS_MEAN = 10
CHUNK = 200_000  # events per output chunk

# results are appended chunk by chunk, so never start on top of an old run
if exists(output_file) or exists(output_file + ".gz"):
    sys.exit(f"{output_file}(.gz) already exists; delete it before running.")
print(f"Language set: {LANG}, window: {WINDOW_HOURS}h")

''' STEP 1: Load users and all their tweets (once) '''
ids = pd.read_csv(join(src, "authors_filtered.csv"), usecols=["author_id"])
user_ids = ids["author_id"].astype(int).tolist()
if n_users is not None:
    user_ids = pd.Series(user_ids).sample(n=n_users, random_state=63).tolist()
print(f"Processing {len(user_ids):,} users")

tweets = (
    pl.scan_csv(join(src, "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"),
                low_memory=True, schema_overrides={"lang": pl.Utf8})
    .filter(pl.col("author_id").is_in(user_ids))
    .select(["author_id", "id", "created_at", "lang", "Score"])
    .with_columns(pl.col("created_at").str.to_datetime(time_zone="UTC"))
    .filter((pl.col("created_at") >= pl.datetime(2019, 1, 1, time_zone="UTC")) &
            (pl.col("created_at") < pl.datetime(2023, 4, 1, time_zone="UTC")))
)
emotion_scores = (
    pl.scan_csv(join(src, "./emotions/emotion_inference.csv.gz"))
    .select(["id"] + [f"{e}_v2" for e in EMOTIONS])
    .with_columns([pl.col("id").cast(pl.Int64)] +
                  [pl.col(f"{e}_v2").cast(pl.Float64) for e in EMOTIONS])
)
tw = tweets.join(emotion_scores, on="id", how="left")
if LANG_SETS[LANG] is not None:
    in_lang = pl.col("lang").is_in(LANG_SETS[LANG]).fill_null(False)
    tw = tw.with_columns([pl.when(in_lang).then(pl.col(f"{e}_v2")).otherwise(None)
                          .alias(f"{e}_v2") for e in EMOTIONS])
print("Loading tweets...")
tw = (tw.drop("lang")
        .collect(streaming=True)
        .sort(["author_id", "created_at"], maintain_order=True))
print(f"  {tw.height:,} tweets")

''' STEP 2: Sort key (user, time) and cumulative sums '''
# key = user index * 2^47 + microseconds since 2019-01-01 (fits in int64)
tw = tw.with_columns(
    pl.col("author_id").rank("dense").cast(pl.Int64).alias("uidx"),
    (pl.col("created_at") - pl.datetime(2019, 1, 1, time_zone="UTC"))
        .dt.total_microseconds().alias("t_us"))
assert tw["t_us"].max() < 2 ** 47 and tw["uidx"].max() < 2 ** 15
key = (tw["uidx"].to_numpy() << 47) + tw["t_us"].to_numpy()
assert np.all(np.diff(key) >= 0)

vals, ok, csum, ccnt = {}, {}, {}, {}
for e in EMOTIONS:
    v = tw[f"{e}_v2"].to_numpy(allow_copy=True).astype(float)  # nulls -> nan
    ok[e] = ~np.isnan(v)
    vals[e] = v
    ccnt[e] = np.concatenate([[0], np.cumsum(ok[e])])

''' STEP 3: Events and windows '''
ev = (tw.with_row_index("pos")
        .filter(pl.col("Score").is_not_null())
        .unique(subset=["id"], keep="first", maintain_order=True))
print(f"  {ev.height:,} news events")
ek = key[ev["pos"].to_numpy()]
W = WINDOW_HOURS * 3600 * 10 ** 6
b_lo = np.searchsorted(key, ek - W, side="left")   # created_at >= t - h
b_hi = np.searchsorted(key, ek, side="left")       # created_at <  t
a_lo = np.searchsorted(key, ek, side="right")      # created_at >  t
a_hi = np.searchsorted(key, ek + W, side="right")  # created_at <= t + h


def window_stats(lo, hi, side):
    out = {f"n_tweets_{side}": hi - lo}
    for e in EMOTIONS:
        out[f"n_{e}_{side}"] = ccnt[e][hi] - ccnt[e][lo]
    return out


bs, as_ = window_stats(b_lo, b_hi, "before"), window_stats(a_lo, a_hi, "after")
keep = np.ones(ev.height, dtype=bool)
for st in (bs, as_):
    for e in EMOTIONS:
        keep &= st[f"n_{e}_{'before' if st is bs else 'after'}"] > 0
print(f"  {keep.sum():,} events with classified tweets before and after")


def mean_and_list(e, lo, hi):
    v = vals[e][lo:hi]
    v = v[~np.isnan(v)]
    m = float(v.mean()) if len(v) >= MIN_TWEETS_MEAN else float("nan")
    return m, " ".join(map(str, v.tolist()))


''' STEP 4: Assemble and write in chunks '''
idx = np.flatnonzero(keep)
ev_kept = ev[idx]
for start in tqdm(range(0, len(idx), CHUNK), desc="Writing chunks"):
    sl = slice(start, start + CHUNK)
    ii = idx[sl]
    e_df = ev_kept[sl]
    cols = {
        "event_id": e_df["id"],
        "author_id": e_df["author_id"],
        "event_timestamp": e_df["created_at"],
        "event_score": e_df["Score"],
    }
    lists = {}
    for side, lo_arr, hi_arr in (("before", b_lo, b_hi), ("after", a_lo, a_hi)):
        st = bs if side == "before" else as_
        cols[f"n_tweets_{side}"] = st[f"n_tweets_{side}"][ii]
        for e in EMOTIONS:
            ml = [mean_and_list(e, lo, hi) for lo, hi in zip(lo_arr[ii], hi_arr[ii])]
            lists[(e, side)] = ml
    # same column order as the loop version
    order = []
    for side in ("before", "after"):
        st = bs if side == "before" else as_
        cols[f"n_anger_{side}"] = st[f"n_anger_{side}"][ii]
        cols[f"n_fear_{side}"] = st[f"n_fear_{side}"][ii]
        cols[f"anger_{side}_list"] = [x[1] for x in lists[("anger", side)]]
        cols[f"fear_{side}_list"] = [x[1] for x in lists[("fear", side)]]
        cols[f"mean_anger_{side}"] = [x[0] for x in lists[("anger", side)]]
        cols[f"mean_fear_{side}"] = [x[0] for x in lists[("fear", side)]]
        cols[f"n_disgust_{side}"] = st[f"n_disgust_{side}"][ii]
        cols[f"disgust_{side}_list"] = [x[1] for x in lists[("disgust", side)]]
        cols[f"mean_disgust_{side}"] = [x[0] for x in lists[("disgust", side)]]
        order += [f"n_tweets_{side}", f"n_anger_{side}", f"n_fear_{side}",
                  f"anger_{side}_list", f"fear_{side}_list",
                  f"mean_anger_{side}", f"mean_fear_{side}",
                  f"n_disgust_{side}", f"disgust_{side}_list", f"mean_disgust_{side}"]
    for e in EMOTIONS:
        cols[f"event_{e}"] = e_df[f"{e}_v2"]
    df_chunk = pl.DataFrame(cols).select(
        ["event_id", "author_id", "event_timestamp", "event_score"] + order +
        [f"event_{e}" for e in EMOTIONS])
    if start == 0:
        df_chunk.write_csv(output_file)
    else:
        with open(output_file, "ab") as f:
            df_chunk.write_csv(f, include_header=False)

''' STEP 5: Compress '''
print("\nCompressing final results...")
with open(output_file, "rb") as f_in, gzip.open(output_file + ".gz", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
os.remove(output_file)
print("Done!")
