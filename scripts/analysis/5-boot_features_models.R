# ===== SCRIPT OVERVIEW =====
# Task:   Bootstrap logistic regression models testing whether user-level emotion
#         dynamics features (baseline, variability, instability, inertia) predict
#         news sharing, misinformation sharing, and partisan sharing.
#         All predictors are z-normalised; R=1000 bootstrap replications, 20 cores.
# Input:  data/Austria-Panel-Users-Features-Tweet.csv
# Output: users/user_models_tweet_boot.rds  (list of boot objects for all model variants)
# Usage:  Rscript 5-boot_features_models.R

library(readr)
library(dplyr)
library(boot)

R <- 10 # test bootstrap samples
n_cores <- 20
src <- "../../data"
dst <- "../../output"
file <- "Austria-Panel-Users-Features-Tweet.csv"

features <- read_csv(file.path(src, file),
                     col_types = cols(
                       .default = col_double()
                     )) 


features$shares_news <- as.integer(features$n_news_shared > 0)
features$shares_untrustworthy <- as.integer(features$n_untrustworthy_shared > 0)
features$shares_biased <- as.integer(features$n_biased_shared > 0)


complete_features <- features[complete.cases(features[, 
                        c('shares_untrustworthy',  # shouldnt have NaNs but just in case
                          'shares_news',
                          'shares_biased',
                          'anger_baseline', 
                          'fear_baseline',
                          'anger_variability',  # everything after this
                          'fear_variability', # could have NaNs
                          'anger_instability',  
                          'fear_instability',
                          'anger_inertia', 
                          'fear_inertia')]), ]

z_features <- complete_features %>%
  mutate(across(c(anger_baseline, fear_baseline, 
                  anger_variability, fear_variability,
                  anger_instability, fear_instability,
                  anger_inertia, fear_inertia),
                ~as.numeric(scale(.))))

df <- z_features

boot_stat <- function(data, indices, formula) {
  d <- data[indices, ]
  model <- glm(formula, data = d, family = binomial)
  return(coef(model))
}

# Bootstrap for NEWS models
print("Running bootstrap for NEWS models...")
boot_news_base <- boot(data = df,
                       statistic = boot_stat,
                       R = R,
                       parallel = "multicore", 
                       ncpus = n_cores,     
                       formula = shares_news ~ anger_baseline + fear_baseline)

boot_news_var <- boot(data = df,
                      statistic = boot_stat,
                      R = R,
                      parallel = "multicore", 
                      ncpus = n_cores,    
                      formula = shares_news ~ anger_variability + fear_variability)

boot_news_inst <- boot(data = df,
                       statistic = boot_stat,
                       R = R,
                       parallel = "multicore", 
                       ncpus = n_cores,    
                       formula = shares_news ~ anger_instability + fear_instability)

boot_news_inertia <- boot(data = df,
                          statistic = boot_stat,
                          R = R, 
                          parallel = "multicore", 
                          ncpus = n_cores,
                          formula = shares_news ~ anger_inertia + fear_inertia)

# Bootstrap for UNTRUSTWORTHY models
print("Running bootstrap for UNTRUSTWORTHY models...")
boot_misinfo_base <- boot(data = df,
                          statistic = boot_stat,
                          R = R,
                          parallel = "multicore", 
                          ncpus = n_cores,
                          formula = shares_untrustworthy ~ anger_baseline + fear_baseline)

boot_misinfo_var <- boot(data = df,
                         statistic = boot_stat,
                         R = R,
                         parallel = "multicore", 
                         ncpus = n_cores,
                         formula = shares_untrustworthy ~ anger_variability + fear_variability)

boot_misinfo_inst <- boot(data = df,
                          statistic = boot_stat,
                          R = R,
                          parallel = "multicore", 
                          ncpus = n_cores,
                          formula = shares_untrustworthy ~ anger_instability + fear_instability)

boot_misinfo_inertia <- boot(data = df,
                             statistic = boot_stat,
                             R = R,
                             parallel = "multicore",
                             ncpus = n_cores,
                             formula = shares_untrustworthy ~ anger_inertia + fear_inertia)

# Bootstrap for PARTISAN models
print("Running bootstrap for PARTISAN models...")
boot_partisan_base <- boot(data = df,
                           statistic = boot_stat,
                           R = R,
                           parallel = "multicore", 
                           ncpus = n_cores,
                           formula = shares_biased ~ anger_baseline + fear_baseline)

boot_partisan_var <- boot(data = df,
                          statistic = boot_stat,
                          R = R,
                          parallel = "multicore", 
                          ncpus = n_cores,
                          formula = shares_biased ~ anger_variability + fear_variability)

boot_partisan_inst <- boot(data = df,
                           statistic = boot_stat,
                           R = R,
                           parallel = "multicore", 
                           ncpus = n_cores,
                           formula = shares_biased ~ anger_instability + fear_instability)

boot_partisan_inertia <- boot(data = df,
                              statistic = boot_stat,
                              R = R,
                              parallel = "multicore", 
                              ncpus = n_cores,
                              formula = shares_biased ~ anger_inertia + fear_inertia)


# dump bootstrap results
print("Saving bootstrap results...")
boot_objects <- list(
  boot_news_base = boot_news_base,
  boot_news_var = boot_news_var,
  boot_news_inst = boot_news_inst,
  boot_news_inertia = boot_news_inertia,
  boot_misinfo_base = boot_misinfo_base,
  boot_misinfo_var = boot_misinfo_var,
  boot_misinfo_inst = boot_misinfo_inst,
  boot_misinfo_inertia = boot_misinfo_inertia,
  boot_partisan_base = boot_partisan_base,
  boot_partisan_var = boot_partisan_var,
  boot_partisan_inst = boot_partisan_inst,
  boot_partisan_inertia = boot_partisan_inertia
)

saveRDS(boot_objects, "users/user_models_tweet_boot.rds")
print("Bootstrap analysis completed.")