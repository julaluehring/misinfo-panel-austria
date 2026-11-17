# Task:   Evaluate the classifier on the annotated sample: weighted AUC, precision, recall and F1
#         at 0.5 and 0.65, inter-rater agreement, ROC curves.
# Input:  data/emotions/validation/validation_v2_key.csv
#         <annotated.csv> [second_coder.csv]
# Output: data/emotions/validation/validation_v2_results.csv
#         output/figures/roc_curve_v2*.pdf
# Run:    python 5-evaluate_validation_v2.py ../../../data <annotated.csv> [second_coder.csv]

import os
import sys
from os.path import join, dirname, abspath
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_fscore_support,
                             cohen_kappa_score)

EMOTIONS = ["anger", "fear", "disgust"]


def read(f):
    sep = ";" if open(f, encoding="utf-8").readline().count(";") > 0 else ","
    return pd.read_csv(f, sep=sep, dtype={"id": str})


src = sys.argv[1]
val_dir = join(src, "emotions", "validation")
ann = read(sys.argv[2])
if "uncodable" in ann.columns:
    # tweets marked as uncodable are excluded from all metrics
    unc = ann["uncodable"].astype(str).str.strip().isin(["1", "1.0"])
    print(f"Excluding {unc.sum()} uncodable tweets")
    ann = ann[~unc]
key = pd.read_csv(join(val_dir, "validation_v2_key.csv"), dtype={"id": str})
d = ann[["id"] + [f"{e}_manual" for e in EMOTIONS]].merge(key, on="id", how="inner")
missing = d[[f"{e}_manual" for e in EMOTIONS]].isna().any(axis=1).sum()
print(f"{len(d)} annotated tweets ({missing} with missing labels are dropped per emotion)")

rows = []
for lang, g in [("de+en", d)] + list(d.groupby("lang")):
    for e in EMOTIONS:
        s = g.dropna(subset=[f"{e}_manual"])
        # weights N_stratum / n_stratum (per language, so pooled rows reflect
        # the population of German and English classified tweets)
        y, p, w = s[f"{e}_manual"].astype(int), s[f"{e}_v2"], s["weight"]
        res = {"lang": lang, "emotion": e, "n": len(s), "positives": int(y.sum())}
        for tag, ww in [("w", w), ("unw", None)]:
            res[f"auc_{tag}"] = roc_auc_score(y, p, sample_weight=ww) if y.nunique() == 2 else np.nan
            for thr, sfx in [(0.5, ""), (0.65, "_065")]:
                pr, rc, f1, _ = precision_recall_fscore_support(
                    y, (p >= thr).astype(int), average="binary", sample_weight=ww,
                    zero_division=0)
                res.update({f"precision{sfx}_{tag}": pr, f"recall{sfx}_{tag}": rc,
                            f"f1{sfx}_{tag}": f1})
        rows.append(res)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(2).to_string(index=False))
out.to_csv(join(val_dir, "validation_v2_results.csv"), index=False)

# ROC curves (as in the first-submission validation notebook): one panel per language, AUC in
# the legend. Unweighted curves describe the annotated sample; weighted curves
# (population of classified tweets) are step-like because a few positives from
# the large strata carry most of the weight. Each with and without disgust.
emotions_cmap = {"anger": "#e41a1c", "fear": "#377eb8", "disgust": "#e69f00"}
lang_labels = {"de": "German", "en": "English"}
# figures go to <repo>/output/figures
dst = join(dirname(abspath(__file__)), "..", "..", "..", "output", "figures")
os.makedirs(dst, exist_ok=True)


def plot_roc(emotions, weighted, fname):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=True)
    for ax, (lang, g) in zip(axes, d.groupby("lang")):
        for e in emotions:
            s_ = g.dropna(subset=[f"{e}_manual"])
            y, p = s_[f"{e}_manual"].astype(int), s_[f"{e}_v2"]
            w = s_["weight"] if weighted else None
            fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
            ax.plot(fpr, tpr, color=emotions_cmap[e], lw=1,
                    label=f"{e} ({roc_auc_score(y, p, sample_weight=w):.2f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(lang_labels.get(lang, lang), fontsize=8)
        ax.set_xlabel("False Positive Rate", fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.legend(loc="lower right", frameon=False, fontsize=7)
    axes[0].set_ylabel("True Positive Rate", fontsize=7)
    plt.tight_layout()
    plt.savefig(join(dst, fname), dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")


no_disgust = [e for e in EMOTIONS if e != "disgust"]
plot_roc(EMOTIONS, False, "roc_curve_v2.pdf")
plot_roc(no_disgust, False, "roc_curve_v2_no_disgust.pdf")
plot_roc(EMOTIONS, True, "roc_curve_v2_weighted.pdf")
plot_roc(no_disgust, True, "roc_curve_v2_weighted_no_disgust.pdf")

if len(sys.argv) > 3:
    sec = read(sys.argv[3])
    both = ann.merge(sec, on="id", suffixes=("_a", "_b"))
    print(f"\nInter-rater agreement on {len(both)} double-coded tweets:")
    try:
        import krippendorff
    except ImportError:
        krippendorff = None
    for e in EMOTIONS:
        a, b = both[f"{e}_manual_a"], both[f"{e}_manual_b"]
        ok = a.notna() & b.notna()
        kappa = cohen_kappa_score(a[ok].astype(int), b[ok].astype(int))
        alpha = (krippendorff.alpha(reliability_data=[a[ok].astype(int), b[ok].astype(int)],
                                    level_of_measurement="nominal")
                 if krippendorff else np.nan)
        print(f"  {e}: kappa = {kappa:.2f}, alpha = {alpha:.2f}, "
              f"agreement = {(a[ok] == b[ok]).mean():.2f}")
