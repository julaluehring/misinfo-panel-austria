# ===== SCRIPT OVERVIEW =====
# Task:   Fit VECM models on the daily time series and bootstrap Impulse Response
#         Functions (IRF) for three variable sets: (fear, anger, score),
#         (fear, anger, news), (fear, anger, tweet count).
# Input:  <DATA_DIR>/Austria-Panel-Daily.csv
# Output: <IRF_DIR>/*.csv — bootstrapped IRF confidence intervals per model
# Usage:  Rscript 3b-boot_irf.R [DATA_DIR] [IRF_DIR]
#         (defaults to data/ and data/IRF/)

library(tsDyn)
library(dplyr)
library(lubridate)
library(pbapply)

args <- commandArgs(trailingOnly = TRUE)
src <- "../../data"
dst <- "./IRF"
dir.create(dst, showWarnings = FALSE, recursive = TRUE)
df <- read.csv(file.path(src, "Austria-Panel-Daily.csv"))
n_boot <- as.numeric(args[1])

# transform data
df$day <- as.Date(df$day, format = "%Y-%m-%d")
daily_summary <- df %>% arrange(day)
daily_summary$mean_fear <- daily_summary$mean_fear * 100
daily_summary$mean_anger <- daily_summary$mean_anger* 100


# create time series data
create_ts <- function(data, var_name) {
  start_date <- min(data$day)
  ts(data[[var_name]], 
     start = c(year(start_date), yday(start_date)), 
     frequency = 7)
}


start_date <- min(daily_summary$day)
start_year <- year(start_date)
start_day  <- yday(start_date)  # day of year for start

ts_vars <- c("mean_fear", "mean_anger", 
             "mean_score", "count_ng_domains", 
             "tweet_count")

ts_list <- setNames(lapply(ts_vars, function(x) create_ts(daily_summary, x)), 
                    c("ts_fear", "ts_anger",
                      "ts_score", "ts_news", 
                      "ts_count")
                    )

score_matrix <- cbind(ts_list$ts_fear, ts_list$ts_anger, ts_list$ts_score)
colnames(score_matrix) <- c("fear", "anger", "score")

news_matrix <- cbind(ts_list$ts_fear, ts_list$ts_anger, ts_list$ts_news)  
colnames(news_matrix) <- c("fear", "anger", "news")

count_matrix <- cbind(ts_list$ts_anger, ts_list$ts_fear, ts_list$ts_count)
colnames(count_matrix) <- c("fear", "anger", "count")

# fit VECM
vecm_models <- list(
  score_emo = VECM(score_matrix, lag = 7, r = 2),
  news_emo = VECM(news_matrix, lag = 9, r = 2),
  count_emo = VECM(count_matrix, lag = 8, r = 2)
)

# define function to bootstrap irf
determine_irf <- function(vecm_model, 
                          impulse_vars = NULL, 
                          response_vars = NULL,
                          max_horizon = 180,
                          boot_runs = n_boot) {
  
  irf_full <- irf(vecm_model, 
                  impulse = impulse_vars,
                  response = response_vars,
                  n.ahead = max_horizon, 
                  boot = TRUE,
                  ci = 0.95,
                  bias.corr = TRUE,
                  runs = boot_runs)
  
  # find last significant period
  significance_results <- list()
  
  for(imp in names(irf_full$irf)) {
    for(resp in colnames(irf_full$irf[[imp]])) {
      
      periods <- 1:max_horizon
      significant_periods <- c()
      
      for(p in periods) {
        lower_bound <- irf_full$Lower[[imp]][p, resp]
        upper_bound <- irf_full$Upper[[imp]][p, resp]
        is_significant <- (lower_bound > 0 & upper_bound > 0) | (lower_bound < 0 & upper_bound < 0)
        
        if(is_significant) {
          significant_periods <- c(significant_periods, p)
        }
      }
      
      # find horizon: last significant + buffer 
      last_significant <- ifelse(length(significant_periods) > 0, 
                                 max(significant_periods), 
                                 0)
      
      significance_results[[paste(imp, resp, sep = "_")]] <- list(
        impulse = imp,
        response = resp,
        last_significant_period = last_significant,
        total_significant_periods = length(significant_periods)
      )
    }
  }
  
  significance_df <- do.call(rbind, lapply(significance_results, function(x) {
    data.frame(
      impulse = x$impulse,
      response = x$response,
      last_significant_period = x$last_significant_period,
      total_significant_periods = x$total_significant_periods
    )
  }))
  
  
  all_results <- data.frame()
  
  for(imp in names(irf_full$irf)) {
    for(resp in colnames(irf_full$irf[[imp]])) {
      temp_df <- data.frame(
        shock = imp,
        response = resp,
        period = 1:nrow(irf_full$irf[[imp]]),
        irf_response = irf_full$irf[[imp]][, resp],
        lower = irf_full$Lower[[imp]][, resp],
        upper = irf_full$Upper[[imp]][, resp]
      )
      all_results <- rbind(all_results, temp_df)
    }
  }
  
  filename_regular <- paste0(response_vars, "_irf_results.csv")
  filename_significance <- paste0(response_vars, "_irf_sig_periods.csv")
  
  write.csv(all_results, file.path(dst, filename_regular), row.names = FALSE)
  write.csv(significance_df, file.path(dst, filename_significance), row.names = FALSE)
  
  return(list(
    results = all_results,
    significance = significance_df
  ))
}

model_specs <- list(
  score_emo = list(impulse = c("ts_fear", "ts_anger"), response = "ts_score"),
  news_emo = list(impulse = c("ts_fear", "ts_anger"), response = "ts_news"),
  count_emo = list(impulse = c("ts_anger", "ts_fear"), response = "ts_count")
)

total_models <- length(vecm_models)
irf_results <- list()

for(i in seq_along(vecm_models)) {
  model_name <- names(vecm_models)[i]
  
  cat(sprintf("\n[%d/%d] Processing %s model...\n", i, total_models, model_name))
  
  start_time <- Sys.time()
  
  irf_results[[model_name]] <- determine_irf(
    vecm_model = vecm_models[[model_name]],
    impulse_vars = model_specs[[model_name]]$impulse,
    response_vars = model_specs[[model_name]]$response,
    max_horizon = 180,
    boot_runs = n_boot
  )
  
  end_time <- Sys.time()
  cat(sprintf("Completed %s in %.1f minutes\n", model_name, as.numeric(end_time - start_time, units = "mins")))
  cat(paste(rep("=", 50), collapse=""), "\n")
}

cat("\nAll models completed!\n")