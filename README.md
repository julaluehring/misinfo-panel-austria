# Emotions and Misinformation in the Austrian Twitter Panel

Public repository for the data collection and analysis code to reproduce the statistical analysis. After downloading this repository, place the `data/` folder into the main directory (i.e., on the same level as the `scripts/` folder). We provide an `environment.yml` to re-create the conda environment (`conda env create -f environment.yml`) and a `requirements.txt` for pip-based installation.

## How to reproduce

To reproduce the **main time-series results** (Figures 1, S2–S6, S11; Tables S2–S14), download and unzip emomis-austria-data.tar.gz, and place data/ at the same level as scripts/ (see repository tree below). Then, run scripts `2-describe_time_series.Rmd`, `3a-test_var.Rmd`, and `3b-boot_irf.R` from `scripts/analysis/`. 

To reproduce the **user-level results** (Figures 2, S10, S12–S17; Tables S15–S22), place the anonymized files from `release/` in `data/` and run scripts `5-boot_features_models.R` through `13-analyze_cohorts.Rmd` in order. 

All figures and tables are written to `output/`. Scripts are designed to be run from `scripts/analysis/` as the working directory, or knitted directly from an RStudio project at the repository root.

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
data/                 All input and intermediate data files (not committed to git)
output/               Figures and tables (not committed to git)
```

## Reproduction of analysis

The repository is organized into four stages: `data collection`, `data_preparation`, `inference`, and all statistical analyses and visualizations in `analysis`.

- `data_preparation/` and `inference/` scripts handle data collection and wrangling. **Most of these steps cannot be reproduced** without access to the raw data or the APIs (see Restrictions).
- The `analysis/` scripts can be reproduced starting from `Austria-Panel-Daily.csv` (time series analysis) or from anonymized user-level feature files (user-level analysis). See the restrictions below.

### Restrictions

There are three major restrictions to fully reproducing the study:

1. **Raw tweet text and user IDs:** Due to Twitter's Terms of Service and GDPR considerations, we cannot publish raw text. We provide anonymized datasets with numeric IDs replaced by sequential integers and all raw text removed. The emotion scores and domain-level aggregates are included in the released data.

2. **NewsGuard scores:** The NewsGuard database is proprietary. We cannot publish domain ratings or names. The analysis uses aggregated daily statistics derived from these ratings (`Austria-Panel-Daily.csv`), which are safe to release. The trustworthiness inference pipeline cannot be reproduced without a NewsGuard license.

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
- `8-anonymize_ids.py`: replace real `author_id` and tweet `event_id` values with randomized sequential integers across all releasable files (`authors_filtered.csv`, `Austria-Panel-Users-Features-Tweet.csv`, `Austria-Panel-Users-Daily.csv.gz`, `Austria-Panel-User-Statistics.csv`, `Austria-Panel-Supersharers.csv`, `Austria-Panel-Tweets-Windows-1h-10min.csv.gz`, `Austria-Panel-Tweets-Windows-24h-10min.csv.gz`); output is gzip-compressed --> `release/` (mapping is not saved and cannot be reversed).

## Inference

### Emotions

Emotion inference uses the fine-tuned mDeBERTa model `pol_emo_mDeBERTa` (stored in `scripts/inference/emotions/pol_emo_mDeBERTa/`, excluded from git, but find the model [here](https://github.com/tweedmann/pol_emo_mDeBERTa2)). It returns a probability (0-1) of a tweet containing one of 8 emotions: anger, fear, disgust, sadness, joy, enthusiasm, pride, hope.

- `1-prepare_text.py`: clean tweet text (remove emojis, URLs, mentions, retweets) --> `emotions/Austria-Panel-Text.csv.gz`
- `2-split_files.py`: split into 30M-row chunks for parallel GPU inference --> `emotions/split_files/`
- `3-infer_emotion_split.py`: apply mDeBERTa model --> `emotions/emotion_inference.csv.gz`

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

- `1-aggregate_data.py`: aggregate to a daily time series --> `Austria-Panel-Daily.csv` 

______

**From here on the main analysis steps are reproducible!**
- `2-describe_time_series.Rmd`: univariate series description, summary statistics, and main time series figures
- `3a-test_var.Rmd`: VAR model diagnostics on the daily time series (lag selection, Granger causality, residual checks)
- `3b-boot_irf.R`: bootstrap Impulse Response Functions from the VAR model → `data/IRF/`
- `4-extract_features.py`: compute per-user emotion dynamics features (baseline, variability, instability, inertia) and news-sharing counts → `Austria-Panel-Users-Features-Tweet.csv` and `Austria-Panel-Users-Daily.csv.gz`
- `5-boot_features_models.R`: bootstrap logistic/linear regressions testing whether emotion features predict news sharing → `scripts/analysis/users/user_models_tweet_boot.rds`
- `6-analyze_features.Rmd`: user emotion feature distributions and bootstrap regression results
- `7-aggregate_windows.py`: extract tweet windows before and after news-sharing events (±1h and ±24h) → `Austria-Panel-Windows-{1h,24h}-10min.csv.gz`
- `8-boot_window_models.R`: bootstrap LMER models testing whether untrustworthy news exposure affects subsequent emotion → `scripts/analysis/windows_models/windows_10_{1h,24h}/`
- `9-analyze_windows.Rmd`: event window comparisons and LMER model results
- `10-aggregate_user_stats.py`: compute per-user summary statistics → `Austria-Panel-User-Statistics.csv`
- `11-analyze_supersharers.Rmd`: identify misinformation supersharers → `Austria-Panel-Supersharers.csv` (required by `12c-pull_sources.py`)
- `12a-pull_replies.py`: build the full reply network → `Austria-Panel-Network-Edges.csv.gz`, `Austria-Panel-Network-Nodes.csv.gz`
- `12b-pull_replies_internal.py`: build the internal reply network (sample-only edges) → `Austria-Panel-Network-Edges-Internal.csv`, `Austria-Panel-Network-Nodes-Internal.csv`
- `12c-pull_sources.py`: build the domain co-sharing network among supersharers → `Austria-Panel-Source-Edges.csv`, `Austria-Panel-Source-Nodes.csv`
- `12d-analyze_replies.Rmd`: reply network analysis → `output/figures/edge_node_distributions.pdf`, `output/figures/combined_supersharer_networks.png`, `output/tables/network_comparison.tex`
- `13-analyze_cohorts.Rmd`: cohort comparison (new vs. existing accounts around 2022 structural breaks) → `output/figures/user_joins.pdf`
- `14a-pull_hashtags.py`: aggregate tweet-level hashtags to a daily list → `Austria-Panel-Daily-Hashtags.csv`
- `14b-analyze_hashtags.Rmd`: join top hashtags onto outlier dates → `output/tables/outliers.tex`
- `15-analyze_domains.Rmd`: domain trustworthiness distribution and changes over time (Jun–Oct 2022)


All Rmd scripts write figures and tables to `output/`.

## Reproducibility boundary

**Fully reproducible from `Austria-Panel-Daily.csv`** (no raw text, no user IDs, no direct NewsGuard scores): `2-describe_time_series.Rmd`, `3a-test_var.Rmd`, `3b-boot_irf.R`.

**Reproducible from anonymized user features** (requires files from `release/`): `4-extract_features.py`, `5-boot_features_models.R`, `6-analyze_features.Rmd`, `7-aggregate_windows.py`, `8-boot_window_models.R`, `9-analyze_windows.Rmd`, `10-aggregate_user_stats.py`, `11-analyze_supersharers.Rmd`, `13-analyze_cohorts.Rmd`.

**Not reproducible without restricted data:** the inference pipeline requires raw tweet text and the full NewsGuard database; network analyses (`12a-pull_replies.py`, `12d-analyze_replies.Rmd`) require the reply structure with user IDs, which cannot be released due to Twitter's Terms of Service.

## Figures

All figures are written to `output/`. The table below maps each output file to the script that generates it.

| Paper | Output file | Script | Description |
|---|---|---|---|
| **Fig. 1** | `desc_prop_plot.pdf` | `2-describe_time_series.Rmd` | Weekly tweet volume (a) and stacked trustworthy/untrustworthy news proportions (b) with event markers |
| **Fig. 2a** | `features_boot_results.pdf` | `6-analyze_features.Rmd` | Odds ratios (bootstrapped 95% CIs) for emotion features predicting any, partisan, and untrustworthy news sharing |
| **Fig. 2b** | `supersharers_cdf.pdf` | `11-analyze_supersharers.Rmd` | CDF of news sharing concentration — 0.5% of users account for 80% of untrustworthy shares |
| **Fig. 2c** | `supersharers_emotion_comparison.pdf` | `11-analyze_supersharers.Rmd` | Anger baseline, instability, and variability distributions for trustworthy vs. untrustworthy news sharers |
| Fig. S1 | *(ROC curves)* | not in repo | AUC curves for anger (0.85) and fear (0.84) emotion classifier validation — requires annotation data |
| **Fig. S2** | `vars_histograms.pdf` | `2-describe_time_series.Rmd` | Histograms of all daily proportion variables used in the analysis |
| **Fig. S3** | `desc_daily.pdf` | `2-describe_time_series.Rmd` | Daily tweet count and news-link count time series |
| **Fig. S4** | `trustworthiness_daily.pdf` | `2-describe_time_series.Rmd` | Daily avg. NewsGuard score and proportion of untrustworthy links |
| **Fig. S5** | `emotions_daily.pdf` | `2-describe_time_series.Rmd` | Daily proportion of angry and fearful tweets |
| **Fig. S6** | `score_intervention.pdf` | `2-describe_time_series.Rmd` | ARIMA intervention analysis: two structural breaks in trustworthiness (Jul and Aug 2022) |
| Fig. S7 | *(newsguard monthly avg)* | not in repo | Average NewsGuard rating per month in the database — requires proprietary NewsGuard data |
| **Fig. S8** | `domains_classification_prop.pdf` | `15-analyze_domains.Rmd` | Classification of untrustworthy domains (stable/downrated/first appearance) Jun–Oct 2022 |
| **Fig. S9** | `untrustworthy_domains_spikes.pdf` | `15-analyze_domains.Rmd` | Daily count of untrustworthy domains shared during the unstable period, with spike markers |
| **Fig. S10** | `user_joins.pdf` | `13-analyze_cohorts.Rmd` | New vs. old account activity and untrustworthy sharing during 2022 structural breaks |
| **Fig. S11** | `irf_plot.pdf` | `3a-test_var.Rmd` | VAR impulse response functions: response of misinformation, anger, and fear to a 10% shock in each variable |
| **Fig. S12** | `features_hist.pdf` | `6-analyze_features.Rmd` | Histograms of all eight user-level emotion features before transformation |
| **Fig. S13** | `pca_scree_plot.pdf` | `6-analyze_features.Rmd` | PCA scree plot: three components explain 87.1% of variance in emotion features |
| **Fig. S14** | `window_tweet_distributions.pdf` | `9-analyze_windows.Rmd` | Distribution of tweets per user in 1h and 24h windows before/after a news-sharing event |
| **Fig. S15** | `boot_means_comparison.pdf` | `9-analyze_windows.Rmd` | Sensitivity analysis: stability of mean anger/fear estimates as a function of tweets sampled |
| **Fig. S16** | `emotions_models_compared.pdf` | `9-analyze_windows.Rmd` | Comparison of OLS (lm) and mixed-effects (lmer) coefficients for 1h and 24h windows |
| **Fig. S17** | `windows_emotions_coefs.pdf` | `9-analyze_windows.Rmd` | Bootstrap coefficients: anger and fear before, after, and change around news-sharing events (1h and 24h) |
| **Fig. S18** | `combined_supersharer_networks.png` | `12d-analyze_replies.Rmd` | Internal reply networks for untrustworthy vs. trustworthy news supersharers |
| **Fig. S19** | `edge_node_distributions.pdf` | `12d-analyze_replies.Rmd` | Reply edge distribution and degree distributions inside vs. outside the sample |

## Tables

All LaTeX tables are written to `output/`. Main text Table 1 and supplementary Table S1 are manually written.

| Paper | Output file | Script | Description |
|---|---|---|---|
| Table 1 | — | manually written | Overview of user-level emotion dynamic variables |
| Table S1 | — | manually written | Summary of ARIMA dynamics across all four variables |
| **Table S2** | `arima_count.tex` | `2-describe_time_series.Rmd` | ARIMA model for tweet activity |
| **Table S3** | `outliers.tex` | `14b-analyze_hashtags.Rmd` | Detected outliers in tweet activity with top hashtags |
| **Table S4** | `arima_news.tex` | `2-describe_time_series.Rmd` | ARIMA model for news sharing |
| **Table S5** | `arima_anger.tex` | `2-describe_time_series.Rmd` | ARIMA model for anger expression |
| **Table S6** | `arima_fear.tex` | `2-describe_time_series.Rmd` | ARIMA model for fear expression |
| **Table S7** | `score_intervention.tex` | `2-describe_time_series.Rmd` | ARIMA intervention model for trustworthiness (two structural breaks) |
| **Table S8** | `correlation_table.tex` | `3a-test_var.Rmd` | Correlation matrix between all time series variables |
| **Table S9** | `unit_root_table.tex` | `3a-test_var.Rmd` | Unit root tests (ADF, PP, KPSS) for all series |
| **Table S10** | `jo_results_table.tex` | `3a-test_var.Rmd` | Johansen cointegration test results |
| **Table S11** | `diagnostics_table.tex` | `3a-test_var.Rmd` | VAR model diagnostics (lag selection) |
| **Table S12** | `var_summary_table.tex` | `3a-test_var.Rmd` | VAR model summary (lag=9) |
| **Table S13** | `granger_table.tex` | `3a-test_var.Rmd` | Granger causality test results |
| **Table S14** | `fevd_table.tex` | `3a-test_var.Rmd` | Forecast error variance decomposition |
| **Table S15** | `user_table.tex` | `6-analyze_features.Rmd` | News sharing behavior of users after dichotomizing outcome variables |
| **Table S16** | `table_news_models.tex` | `6-analyze_features.Rmd` | Logistic regression predicting any news sharing |
| **Table S17** | `table_misinfo_models.tex` | `6-analyze_features.Rmd` | Logistic regression predicting untrustworthy news sharing |
| **Table S18** | `table_partisan_models.tex` | `6-analyze_features.Rmd` | Logistic regression predicting partisan news sharing |
| **Table S19** | `emotions_before.tex` | `9-analyze_windows.Rmd` | LMER: emotion before sharing predicting news trustworthiness |
| **Table S20** | `emotions_after.tex` | `9-analyze_windows.Rmd` | LMER: emotion after sharing predicted by news trustworthiness |
| **Table S21** | `emotions_change.tex` | `9-analyze_windows.Rmd` | LMER: within-person change in emotion around sharing events |
| **Table S22** | `windows_random_effects.tex` | `9-analyze_windows.Rmd` | Random effects structure across all window models |
| **Table S23** | `network_comparison.tex` | `12d-analyze_replies.Rmd` | Network statistics: untrustworthy vs. trustworthy supersharer reply networks |
| **Table S24** | `replies_latex.tex` | `12d-analyze_replies.Rmd` | Descriptive statistics for reply network (inside vs. outside sample) |