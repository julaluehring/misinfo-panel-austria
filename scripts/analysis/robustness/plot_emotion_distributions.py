# Task:   Distributions of the tweet-level anger and fear scores with mean, median, Q3, 0.5 and
#         0.65.
# Input:  data/Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz (not released)
#         data/emotions/emotion_inference.csv.gz (not released)
#         data/authors_filtered.csv
# Output: data/Austria-Panel-Emotion-Distribution-Stats-<lang>.csv
#         output/figures/emotion_distributions[_<lang>].pdf
# Run:    python plot_emotion_distributions.py ../../../data [lang]   # lang: de_en (default), de, all

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
import os
import sys


src = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "de_en"
LANG_SETS = {"de_en": ["de", "en"], "de": ["de"], "all": None}
assert LANG in LANG_SETS, f"lang must be one of {list(LANG_SETS)}"
file = "Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz"
N_BINS = 80      # bins on a log10 scale
LOG_MIN = -5     # scores below 1e-5 are put into the first bin
EMOTIONS = ["anger", "fear", "disgust"]   # disgust: statistics only (CSV), not plotted
PLOT_EMOTIONS = ["anger", "fear"]

# figure goes to <repo>/output/figures
dst = join(dirname(abspath(__file__)), "..", "..", "..", "output", "figures")
os.makedirs(dst, exist_ok=True)

df = (pl.scan_csv(join(src, file),
                  separator=",",
                  quote_char='"',
                  schema_overrides={"lang": pl.Utf8})
        .select(["id", "author_id", "created_at", "lang"])
        .with_columns([
            pl.col("id").cast(pl.Utf8),
            pl.col("author_id").cast(pl.Utf8),
            pl.col("created_at").str.slice(0, 10).alias("day")])
    )

authors = (pl.scan_csv(join(src, "authors_filtered.csv"),
                       separator=",",
                       quote_char='"')
             .select(pl.col("author_id").cast(pl.Utf8))
             .unique())

emotions = (pl.scan_csv(join(src, "emotions", "emotion_inference.csv.gz"),
                        separator=",")
              .select(["id", "anger_v2", "fear_v2", "disgust_v2"])
              .with_columns([
                  pl.col("id").cast(pl.Utf8),
                  pl.col("anger_v2").cast(pl.Float64),
                  pl.col("fear_v2").cast(pl.Float64), 
                  pl.col("disgust_v2").cast(pl.Float64)]))

# scored tweets of the panel in the observation period
scores = (df
          .join(authors, on="author_id", how="inner")
          .filter(pl.col("day") >= "2019-01-01")
          .filter(pl.col("day") < "2023-04-01")
          .join(emotions, on="id", how="inner"))
if LANG_SETS[LANG] is not None:
    scores = scores.filter(pl.col("lang").is_in(LANG_SETS[LANG]))

stats = {}
hists = {}
for emo in EMOTIONS:
    col = pl.col(f"{emo}_v2")
    s = (scores
         .select([
             col.count().alias("n"),
             col.mean().alias("mean"),
             col.median().alias("median"),
             col.quantile(0.75).alias("q3"),
             col.quantile(0.90).alias("q90"),
             (col >= 0.5).mean().alias("share_ge_05"),
             (col >= 0.65).mean().alias("share_ge_065")])
         .collect()
         .row(0, named=True))
    stats[emo] = s
    # binned counts (computed lazily, no need to hold all scores in memory)
    hists[emo] = (scores
                  .select(((col.clip(10 ** LOG_MIN, 1).log10() - LOG_MIN)
                           / (-LOG_MIN) * N_BINS)
                          .floor().clip(0, N_BINS - 1)
                          .cast(pl.Int32).alias("bin"))
                  .group_by("bin")
                  .agg(pl.len().alias("n"))
                  .sort("bin")
                  .collect()
                  .with_columns((pl.col("n") / pl.col("n").sum() * 100)
                                .alias("pct")))

print(f"\n=== Tweet-level emotion scores (classified tweets, {LANG}) ===")
for emo, s in stats.items():
    print(f"{emo}: n = {s['n']:,}, mean = {s['mean']:.4f}, "
          f"median = {s['median']:.4f}, Q3 = {s['q3']:.4f}, "
          f"Q90 = {s['q90']:.4f}, share >= 0.5 = {s['share_ge_05'] * 100:.2f}%, "
          f"share >= 0.65 = {s['share_ge_065'] * 100:.2f}%")

pl.DataFrame([{"emotion": e, **s} for e, s in stats.items()]) \
  .write_csv(join(src, f"Austria-Panel-Emotion-Distribution-Stats-{LANG}.csv"))

''' PLOT '''
colours = {"anger": "#d7301f", "fear": "#2171b5", "disgust": "#e69f00"}
fig, axes = plt.subplots(1, len(PLOT_EMOTIONS), figsize=(6.5, 3.8), sharey=True)
bin_w = -LOG_MIN / N_BINS  # bin width in log10 units

for ax, emo, tag in zip(axes, PLOT_EMOTIONS, ["a", "b"]):
    h = hists[emo]
    left = 10 ** (LOG_MIN + h["bin"].to_numpy() * bin_w)
    right = 10 ** (LOG_MIN + (h["bin"].to_numpy() + 1) * bin_w)
    ax.bar(left, h["pct"].to_numpy(), width=right - left, align="edge",
           color=colours[emo], alpha=0.6, linewidth=0)
    s = stats[emo]
    ax.axvline(s["mean"], color="black", linestyle="--", linewidth=1,
               label=f"Mean ({s['mean']:.3g})")
    ax.axvline(s["median"], color="black", linestyle=":", linewidth=1.2,
               label=f"Median ({s['median']:.3g})")
    ax.axvline(s["q3"], color="black", linestyle="-", linewidth=1.2,
               label=f"Q3 ({s['q3']:.3g})")
    ax.axvline(0.5, color="grey", linestyle="-", linewidth=1.2,
               label=f"0.5 ({s['share_ge_05'] * 100:.1f}%)")
    ax.axvline(0.65, color="grey", linestyle="--", linewidth=1.2,
               label=f"0.65 ({s['share_ge_065'] * 100:.1f}%)")
    ax.set_xscale("log")
    ax.set_xlim(10 ** LOG_MIN, 1)
    ax.set_xlabel(f"{emo} probability (log)", fontsize=9)
    ax.set_title(tag, loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    # legend below each panel so that it does not cover the vertical lines
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)

axes[0].set_ylabel("tweets (%)", fontsize=9)
plt.tight_layout()
out = join(dst, "emotion_distributions.pdf" if LANG == "de_en"
           else f"emotion_distributions_{LANG}.pdf")
plt.savefig(out, bbox_inches="tight")
print(f"\nSaved plot to {out}")
