# Emotions and Misinformation in the Austrian Twitter Panel

Public repository for the data collection and analysis code to reproduce the statistical analysis. After downloading and unzipping the data from [OSF](https://osf.io/3qhcv/overview?view_only=fdd5ac1d11e74af0b8ee618f862d5ddd), place the `data/` folder into the main directory (i.e., on the same level as the `scripts/` folder).

## Software

The results were produced with R 4.3.1 and Python 3.11.

- **R (pinned):** install R 4.3.1, then run `install.packages("renv"); renv::restore()` from the repository root and choose to activate the project when asked. This installs the exact package versions recorded in `renv.lock` (224 CRAN packages, including all dependencies) into a project library. Pandoc is needed to render the `.Rmd` files (it ships with RStudio). On Linux, some packages are built from source and need system libraries, e.g. on Debian/Ubuntu: `sudo apt install libfontconfig1-dev libfreetype6-dev libfribidi-dev libharfbuzz-dev libx11-dev libxml2-dev pandoc`.
- **Python (pinned):** `pip install -r requirements.txt`. Only needed for the scripts that build the released files from restricted data.
- **Conda (quick, not pinned):** `conda env create -f environment.yml` installs R, Python and all packages, but in their current versions, which can differ slightly from the ones we used.

The bootstraps in `5-boot_features_models.R` and `8-boot_window_models.R` run in parallel (20 and 15 cores). With multithreaded BLAS libraries (e.g. OpenBLAS in conda R), set `export OPENBLAS_NUM_THREADS=1` before running them; otherwise each worker starts one thread per core and the machine is heavily oversubscribed. This does not change the results.

## How to reproduce

To reproduce the **main time-series results**, download and unzip the data archive, and place data/ at the same level as scripts/ (see repository tree below). Then run `2-describe_time_series.Rmd`, `2b-plot_time_series.Rmd` and `3a-test_var.Rmd` from `scripts/analysis/`. `3a-test_var.Rmd` also computes the bootstrapped impulse responses and the SI robustness checks.

To reproduce the **user-level results**, place the anonymized files from `data/` and run, in this order: `5-boot_features_models.R`, `6-analyze_features.Rmd`, `8-boot_window_models.R` (once for `1h`, once for `24h`), `11-analyze_supersharers.Rmd`, `9-analyze_windows.Rmd` and `13-analyze_cohorts.Rmd`. Scripts `4`, `7` and `10` produce the released user-level files from tweet-level data and are not part of the reproduction (see Reproducibility boundary).

Figures are written to `output/figures/`, LaTeX tables to `output/tables/`, and bootstrapped user models to `output/users/`. Scripts are designed to be run from `scripts/analysis/` as the working directory, or knitted directly from an RStudio project at the repository root.

See `CHANGELOG.md` for the changes since the first submission (v1.0).

### Main specification and variants

- **Languages:** emotion scores are used for German and English tweets only (`lang = de_en`, the languages the classifier was validated on) so most scripts take a `lang` argument (`de_en` = main, `de`, `all`), and output files carry a matching suffix, e.g. `Austria-Panel-Daily-de_en.csv`.
- **Emotion presence:** a tweet expresses an emotion if its score is ≥ 0.65. 

## Dataset

The Austrian Twitter Panel covers **206 million tweets** posted between 2019-01-01 and 2023-04-01, collected via the Brandwatch API. The panel follows a fixed sample of Austrian Twitter users across the full observation period. The analysis sample is defined by `authors_filtered.csv` (activity and follower filter applied in `6-filter_authors.py`).


## Repository structure

```
scripts/
  data_collection/    Exemplary Brandwatch API query
  data_preparation/   JSON extraction, concatenation, user filtering
  inference/
    emotions/         Text cleaning and emotion inference
    trustworthiness/  URL expansion, domain extraction, NewsGuard matching
  analysis/           Aggregation, feature extraction, statistical modeling, visualization
    robustness/       Robustness checks and reviewer responses (mostly need restricted data)
data/                 All input and intermediate data files (not committed to git)
output/               figures/, tables/, users/ (not committed to git)
```

## Reproduction of analysis

The repository is organized into four stages: `data collection`, `data_preparation`, `inference`, and all statistical analyses and visualizations in `analysis`.

- `data_preparation/` and `inference/` scripts handle data collection and wrangling. **Most of these steps cannot be reproduced** without access to the raw data or the APIs (see Restrictions).
- The `analysis/` scripts can be reproduced starting from `Austria-Panel-Daily-de_en.csv` (time series analysis) or from anonymized user-level feature files (user-level analysis). See the restrictions below.

### Restrictions

There are three major restrictions to fully reproducing the study:

1. **Raw tweet text and user IDs:** Due to Twitter's Terms of Service and GDPR considerations, we cannot publish raw text. We provide anonymized datasets with numeric IDs replaced by sequential integers and all raw text removed. The emotion scores and domain-level aggregates are included in the released data.

2. **NewsGuard scores:** The NewsGuard database is proprietary. We cannot publish domain ratings or names. The analysis uses aggregated daily statistics derived from these ratings (`Austria-Panel-Daily-<lang>.csv`, `Austria-Panel-Daily-Thresholds.csv`), which are safe to release. The trustworthiness inference pipeline cannot be reproduced without a NewsGuard license.

3. **Data collection:** Raw data is collected from the Brandwatch API. A Brandwatch account with access to the relevant project and query IDs is required; these steps cannot be reproduced without API credentials. Due to the closure of the Twitter API, the data cannot be rehydrated. 

## Data collection

The data was collected in two steps: 

- `GetTweetsQuery.sh`: API calls to Brandwatch for a given query ID and date range, appending results to a JSON file. Finds users located in Austria. 
- `processLines.sh`: processes the collected JSON files using `jq`, extracting tweet fields (`author_id`, `conversation_id`, `created_at`, `id`, `in_reply_to_user_id`, `referenced_tweets`, `lang`, `public_metrics`, `text`) --> `Panel-AT-BWdata.csv`

Then, using the Twitter API, we collected all tweets from the selected users. 

## Data preparation

Raw data arrives as gzipped Brandwatch JSON exports. The preparation pipeline extracts tweet and user fields, concatenates per-file CSVs, strips identification columns for publication-ready code, and builds the analysis sample:

- `1-extract_tweet_data.sh`: extract tweet-level fields from JSON using `jq` --> one CSV per input file in `CSVtweets/`
- `4-extract_user_data.sh`: extract user metadata from the same JSON files --> one CSV per input file in `UserCSVfiles/`
- `2-create_dtypes.py`: generate `dtypes.pkl` for consistent column type coercion across all downstream scripts
- `3-concat_tweet_data.py`: concatenate all per-file tweet CSVs --> `Austria-Panel-Tweets.csv.gz`
- `5-concat_user_data.py`: concatenate all per-file user CSVs, deduplicate by `author_id` --> `Austria-Panel-Users.csv.gz`
- `6-filter_authors.py`: apply activity filter (< 20 tweets/day) and follower filter (50–100,000) --> `authors_filtered.csv` (29,291 users)
- `7-remove_user_columns.py`: strip PII columns (`username`, `name`, `location`, etc.) from `Austria-Panel-Tweets.csv.gz` 
- `8-anonymize_ids.py`: replace real `author_id` and tweet `event_id` values with randomized sequential integers across all releasable files (`authors_filtered.csv`, `Austria-Panel-Daily-{de_en,de,all}.csv`, `Austria-Panel-Daily-Thresholds.csv`, `Austria-Panel-Users-Features-Tweet-{de_en,de}.csv`, `Austria-Panel-Users-Daily.csv.gz`, `Austria-Panel-User-Statistics.csv`, `Austria-Panel-Supersharers.csv`, `Austria-Panel-Supersharers-Groups.csv`, `Austria-Panel-Windows-{1h,24h}-10min-de_en.csv.gz`). One mapping is used for all files in a run, so user-level joins still work. The script stops if an ID is missing from the mapping or an identifying column remains. Output is gzip-compressed --> `data/` (mapping is not saved and cannot be reversed).

## Inference

### Emotions

Emotion inference uses the fine-tuned mDeBERTa model `pol_emo_mDeBERTa` (stored in `scripts/inference/emotions/pol_emo_mDeBERTa/`, excluded from git, but find the model [here](https://github.com/tweedmann/pol_emo_mDeBERTa2)). It returns a probability (0-1) of a tweet containing one of 8 emotions: anger, fear, disgust, sadness, joy, enthusiasm, pride, hope.

- `1-prepare_text.py`: clean tweet text (remove emojis, URLs, mentions, retweets) --> `emotions/Austria-Panel-Text.csv.gz`
- `2-split_files.py`: split into 30M-row chunks for parallel GPU inference --> `emotions/split_files/`
- `3-infer_emotion_split.py`: apply mDeBERTa model --> `emotions/emotion_inference.csv.gz`
- `4-sample_validation_v2.py`: draw a stratified German and English annotation sample for validation --> `emotions/validation/`
- `5-evaluate_validation_v2.py`: weighted AUC, precision, recall and F1 (at 0.5 and 0.65), inter-rater agreement and ROC figure

### Trustworthiness

Trustworthiness scores come from matching tweet URLs against the NewsGuard domain rating database. Scores range from 0–100; scores below 60 are classified as untrustworthy.

- `1-extract_urls.py`: extract and normalize domains from tweet URLs, separating shortened (`urls/shortened_urls_r2.csv.gz`) from directly resolved links (`urls/unshortened_urls_r2.csv.gz`)
- `2-remove_platform_links.ipynb`: remove known platform links (e.g. fb.me, youtu.be) from the shortened URL list before expansion
- `3-unravel_urls.py`: expand remaining shortened URLs via async HTTP requests --> per-batch `urls/unraveled_urls_*.csv.gz`
- `4-add_platform_links.ipynb`: concatenate expanded URLs with manually resolved platform links --> `urls/unraveled_urls_all.csv.gz`
- `5-concat_unraveled_urls.py`: concatenate per-batch unraveled URL files and extract final domains --> `urls/unraveled_domains_reduc.csv.gz`
- `6-create_newsguard_set.ipynb`: build the monthly NewsGuard domain rating lookup table --> `newsguard/domains_unique.csv`
- `7-match_domains.py`: match tweet domains against NewsGuard via exact, registered-domain, and fuzzy matching --> `urls/matched_domains_all.csv.gz`
- `8-match_rating.py`: join domain ratings to tweets by domain and month --> `Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz`

`Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz` is the central intermediate file joining tweets, emotion scores, and NewsGuard domain scores. It is the main input for all downstream analysis scripts.

## Analysis

### Aggregation and feature extraction

- `1-aggregate_data.py`: aggregate to a daily time series --> `Austria-Panel-Daily-<lang>.csv`
- `1b-aggregate_thresholds.py`: daily untrustworthy-news counts at NewsGuard cutoffs 40, 50, 60 and 70 --> `Austria-Panel-Daily-Thresholds.csv` (read by `3a-test_var.Rmd`)
- `1c-pull_hashtags.py`: aggregate tweet-level hashtags to a daily list --> `Austria-Panel-Daily-Hashtags.csv` (not released; read by the last section of `2-describe_time_series.Rmd`, which is skipped without it)

______

**From here on the main analysis steps are reproducible**, except the scripts marked *(restricted)*, which build the released files from tweet-level data.
- `2-describe_time_series.Rmd`: univariate series description, summary statistics, ARIMA models (orders from `auto.arima`), outlier table with top hashtags, and main time series figures
- `2b-plot_time_series.Rmd`: descriptive time series figures (Fig. 2, histograms, daily series); the NewsGuard domain plot runs only with the restricted NewsGuard data
- `3a-test_var.Rmd`: VAR in levels (Cholesky order untrustworthy → fear → anger; news share and two break dummies as exogenous variables), wild-bootstrap Granger tests, bootstrapped IRFs and FEVD, and the SI robustness section (German only, all languages, Q3, 0.5, first-submission specification, disgust, 3-day rolling mean, NewsGuard cutoff 50)
- `4-extract_features.py` *(restricted)*: compute per-user emotion dynamics features (baseline, variability, instability, inertia), activity counts (`n_tweets`, `n_scored`) and news-sharing counts → `Austria-Panel-Users-Features-Tweet-<lang>.csv`; for `lang = de_en` also `Austria-Panel-Users-Daily.csv.gz`
- `5-boot_features_models.R`: bootstrap logistic/linear regressions testing whether emotion features predict news sharing → `output/users/user_models_tweet_boot_<lang>[_activity][_disgust].rds`
- `6-analyze_features.Rmd`: user emotion feature distributions and bootstrap regression results (parameters `lang`, `controls`, `disgust`)
- `7-aggregate_windows.py` *(restricted)*: extract tweet windows before and after news-sharing events (±1h and ±24h) → `Austria-Panel-Windows-{1h,24h}-10min-<lang>.csv.gz`
- `8-boot_window_models.R`: bootstrap LMER models testing whether untrustworthy news exposure affects subsequent emotion → `scripts/analysis/windows_models/windows_10_{1h,24h}_<lang>/`
- `9-analyze_windows.Rmd`: event window comparisons and LMER model results
- `10-aggregate_user_stats.py` *(restricted)*: compute per-user summary statistics → `Austria-Panel-User-Statistics.csv`
- `11-analyze_supersharers.Rmd`: identify misinformation supersharers → `Austria-Panel-Supersharers.csv` (required by `12c-pull_sources.py`) and `Austria-Panel-Supersharers-Groups.csv` (groups by news sharing only; required by `12d-analyze_replies.Rmd`)
- `12a-pull_replies.py`: build the full reply network → `Austria-Panel-Network-Edges.csv.gz`, `Austria-Panel-Network-Nodes.csv.gz`
- `12b-pull_replies_internal.py`: build the internal reply network (sample-only edges) → `Austria-Panel-Network-Edges-Internal.csv`, `Austria-Panel-Network-Nodes-Internal.csv`
- `12c-pull_sources.py`: build the domain co-sharing network among supersharers → `Austria-Panel-Source-Edges.csv`, `Austria-Panel-Source-Nodes.csv`
- `12d-analyze_replies.Rmd`: reply network analysis → `output/figures/edge_node_distributions.pdf`, `output/figures/combined_supersharer_networks.png`, `output/tables/network_comparison.tex`
- `13-analyze_cohorts.Rmd`: cohort comparison (new vs. existing accounts around 2022 structural breaks) → `output/figures/user_joins.pdf`
- `15-analyze_domains.Rmd`: domain trustworthiness distribution and changes over time (Jun–Oct 2022)


All Rmd scripts write figures to `output/figures/` and tables to `output/tables/`.

### Robustness checks (`scripts/analysis/robustness/`)

These scripts support SI analyses and the response to reviewers. Most need tweet-level data and cannot be run on the released files (see each script's header).

- `newsguard_robustness.Rmd`: false-content criterion, threshold sensitivity and sources of the 2022 increase (domain- and user-level sections removed for release)
- `extract_untrustworthy_links.py`: extract untrustworthy links for the NewsGuard checks (output contains author IDs and is not released)
- `describe_language.py`: language shares in the panel
- `plot_emotion_distributions.py`: emotion score distributions with mean, median, Q3, 0.5 and 0.65
- `describe_feature_timescales.py`: variance decomposition and daily/weekly/monthly windows for instability and inertia

## Reproducibility boundary

**Fully reproducible from the daily aggregates** (`Austria-Panel-Daily-<lang>.csv`, `Austria-Panel-Daily-Thresholds.csv`; no raw text, no user IDs, no direct NewsGuard scores): `2-describe_time_series.Rmd`, `2b-plot_time_series.Rmd`, `3a-test_var.Rmd`.

**Reproducible from the anonymized user-level files** (released in `data/`): `5-boot_features_models.R`, `6-analyze_features.Rmd`, `8-boot_window_models.R`, `9-analyze_windows.Rmd`, `11-analyze_supersharers.Rmd`, `13-analyze_cohorts.Rmd`.

**Restricted: scripts that create the released user-level files** (`4-extract_features.py`, `7-aggregate_windows.py`, `10-aggregate_user_stats.py`). They do not need tweet text, but they read the tweet-level file `Austria-Panel-Tweets-Domains-PO-Dynamic.csv.gz` (tweet and user IDs, exact timestamps and per-tweet NewsGuard ratings) and the tweet-level emotion scores. This file is not released: per-tweet NewsGuard ratings fall under the NewsGuard licence (Restriction 2), and exact per-user tweet timestamps could be matched against public tweet archives to re-identify users. Their outputs are released in anonymized form (`Users-Features-Tweet`, `Users-Daily`, `Windows`, `User-Statistics`). The same applies to `1-aggregate_data.py`, `1b-aggregate_thresholds.py` and `1c-pull_hashtags.py` for the daily files.

**Not reproducible without restricted data:** the inference pipeline requires raw tweet text and the full NewsGuard database; network analyses (`12a`–`12d`) require the reply structure with user IDs, which cannot be released due to Twitter's Terms of Service. `15-analyze_domains.Rmd` needs NewsGuard domain data. The scripts in `robustness/` and the validation scripts in `inference/emotions/` need tweet-level data.

## Figures

Numbering follows the revised manuscript (`main.tex`, `supplementary.tex`). **Bold** = reproducible from the released data. Figures are written to `output/figures/`.

| Paper | Output file | Script | Description |
|---|---|---|---|
| **Fig. 1** | `individual_plot.pdf` | `9-analyze_windows.Rmd` | Individual-level predictors of news sharing: odds ratios of the emotion features, concentration of news sharing, emotion features of supersharers |
| **Fig. 2** | `desc_prop_plot.pdf` | `2b-plot_time_series.Rmd` | Activity on Austrian Twitter 2019–2023 (weekly averages of daily counts) with event markers |
| **Fig. S1** | `desc_daily.pdf` | `2b-plot_time_series.Rmd` | Daily tweeting activity, news shared and trustworthiness |
| **Fig. S2** | `vars_histograms.pdf` | `2b-plot_time_series.Rmd` | Histograms of the daily variables used in the analysis |
| Fig. S3 | `domains_plot.pdf` | `2b-plot_time_series.Rmd` | Average NewsGuard rating per month — requires the licensed NewsGuard data |
| **Fig. S4** | `trustworthiness_daily.pdf` | `2b-plot_time_series.Rmd` | Daily trustworthiness score and share of untrustworthy news |
| **Fig. S5** | `emotions_daily.pdf` | `2b-plot_time_series.Rmd` | Daily share of angry and fearful tweets |
| Fig. S6 | `emotion_distributions.pdf` | `robustness/plot_emotion_distributions.py` | Distributions of tweet-level anger and fear scores — requires tweet-level data |
| Fig. S7 | `roc_curve_v2_weighted_no_disgust.pdf` | `inference/emotions/5-evaluate_validation_v2.py` | Weighted ROC curves of the classifier validation (German, English) — requires the annotations |
| **Fig. S8** | `features_hist.pdf` | `6-analyze_features.Rmd` | Histograms of the user features before transformation |
| **Fig. S9** | `pca_scree_plot.pdf` | `6-analyze_features.Rmd` | Scree plot of the principal components of the emotion features |
| **Fig. S10** | `window_tweet_distributions.pdf` | `9-analyze_windows.Rmd` | Tweets per user before and after sharing a news tweet |
| **Fig. S11** | `boot_means_comparison.pdf` | `9-analyze_windows.Rmd` | Sensitivity of the mean emotion estimates to the number of tweets sampled |
| **Fig. S12** | `emotions_models_compared.pdf` | `9-analyze_windows.Rmd` | Comparison of the window models (lm vs. lmer, 1h vs. 24h) |
| **Fig. S13** | `windows_emotions_coefs.pdf` | `9-analyze_windows.Rmd` | Emotional expression before and after sharing news |
| Fig. S14 | `combined_supersharer_networks.png` | `12d-analyze_replies.Rmd` | Internal reply networks of untrustworthy and trustworthy news supersharers — requires the reply network |
| Fig. S15 | `edge_node_distributions.pdf` | `12d-analyze_replies.Rmd` | Reply edges by direction and degree distributions — requires the reply network |
| **Fig. S16** | `score_intervention.pdf` | `2-describe_time_series.Rmd` | Two structural breaks in trustworthiness (1 July and 28 August 2022) |
| Fig. S17 | `domains_classification_prop.pdf` | `15-analyze_domains.Rmd` | Classification of the domains shared from July 2022 — requires NewsGuard domain data |
| Fig. S18 | `untrustworthy_domains_spikes.pdf` | `15-analyze_domains.Rmd` | Untrustworthy domains shared from July 2022 with spike days — requires NewsGuard domain data |
| **Fig. S19** | `user_joins.pdf` | `13-analyze_cohorts.Rmd` | Users joining the platform and their untrustworthy news sharing |
| Fig. S20 | `ng_thresholds.pdf` | `robustness/newsguard_robustness.Rmd` | Weekly share of untrustworthy and trustworthy news at different NewsGuard cutoffs — the figure itself only needs `Austria-Panel-Daily-Thresholds.csv`, but earlier sections of the script need the licensed NewsGuard data |
| **Fig. S21** | `irf_plot.pdf` | `3a-test_var.Rmd` | Impulse response functions of the VAR(3) |
| **Fig. S22** | `irf_plot_90d.pdf` | `3a-test_var.Rmd` | Impulse response functions over 90 days |

## Tables

LaTeX tables are written to `output/tables/`. Table 1 (`tables/emotion_measures`, overview of the user-level variables) is written by hand.

| Paper | Output file | Script | Description |
|---|---|---|---|
| Table 1 | — | written by hand | Overview of the user-level variables |
| **Table S1** | `user_table.tex` | `6-analyze_features.Rmd` | News sharing of users after dichotomizing the outcomes |
| **Table S2** | `table_news_models.tex` | `6-analyze_features.Rmd` | Logistic regressions predicting sharing any news |
| **Table S3** | `table_misinfo_models.tex` | `6-analyze_features.Rmd` | Logistic regressions predicting sharing untrustworthy news |
| **Table S4** | `table_partisan_models.tex` | `6-analyze_features.Rmd` | Logistic regressions predicting sharing partisan news |
| **Table S5** | `emotions_before.tex` | `9-analyze_windows.Rmd` | Mixed models: emotional expression before sharing predicting untrustworthy news |
| **Table S6** | `emotions_after.tex` | `9-analyze_windows.Rmd` | Mixed models: emotional expression after sharing |
| **Table S7** | `emotions_change.tex` | `9-analyze_windows.Rmd` | Mixed models: within-person change in emotional expression |
| **Table S8** | `windows_random_effects.tex` | `9-analyze_windows.Rmd` | Random effects of the window models |
| Table S9 | `network_comparison.tex` | `12d-analyze_replies.Rmd` | Reply network statistics of untrustworthy vs. trustworthy news supersharers — requires the reply network |
| Table S10 | `replies_latex.tex` | `12d-analyze_replies.Rmd` | Replies inside vs. outside the sample — requires the reply network |
| **Table S11** | `arima_count.tex` | `2-describe_time_series.Rmd` | ARIMA model for tweet activity |
| Table S12 | `outliers.tex` | `2-describe_time_series.Rmd` | Outliers in tweet activity with the top 3 hashtags — requires the hashtags file, which is not released |
| **Table S13** | `arima_news.tex` | `2-describe_time_series.Rmd` | ARIMA model for news sharing |
| **Table S14** | `arima_anger.tex` | `2-describe_time_series.Rmd` | ARIMA model for anger |
| **Table S15** | `arima_fear.tex` | `2-describe_time_series.Rmd` | ARIMA model for fear |
| **Table S16** | `score_intervention.tex` | `2-describe_time_series.Rmd` | ARIMA model with intervention for trustworthiness |
| **Table S17** | `correlation_matrix.tex` | `3a-test_var.Rmd` | Correlations between all daily variables |
| **Table S18** | `unit_root_table.tex` | `3a-test_var.Rmd` | Unit root tests (ADF, PP, KPSS) |
| **Table S19** | `jo_results_table.tex` | `3a-test_var.Rmd` | Johansen cointegration test (trace) |
| **Table S20** | `diagnostics_table.tex` | `3a-test_var.Rmd` | Diagnostics of VAR models with different lag orders |
| **Table S21** | `var_summary_table.tex` | `3a-test_var.Rmd` | Summary of the VAR in levels (lag order 3) |
| **Table S22** | `granger_table.tex` | `3a-test_var.Rmd` | Granger causality tests |
| **Table S23** | `shock_table.tex` | `3a-test_var.Rmd` | Shock sizes used for the impulse response functions |
| **Table S24** | `fevd_table.tex` | `3a-test_var.Rmd` | Forecast error variance decomposition |
| **Table S25** | `var_robustness_table.tex` | `3a-test_var.Rmd` | Robustness checks of the VAR |

Some scripts write further figures that are not in the paper (e.g. `desc_plot.pdf`, `supersharers_cdf.pdf`, `windows_caterpillar_combined.pdf`). Variant runs add a `_<lang>`, `_activity` or `_disgust` suffix to the file names.
