# Task:   Bootstrapped logistic regressions (1000 replicates): do the emotion features predict
#         sharing news, untrustworthy news and partisan news?
# Input:  data/Austria-Panel-Users-Features-Tweet-<lang>.csv
# Output: output/users/user_models_tweet_boot_<lang>[_activity][_disgust].rds
# Run:     As R script or in terminal as Rscript 5-boot_features_models.R [lang] [controls] [emotions]   # defaults: de_en none main; alternatives: de | activity | disgust

library(readr)
library(dplyr)
library(boot)

args     <- commandArgs(trailingOnly = TRUE)
lang     <- if (length(args) >= 1) args[1] else "de_en"
controls <- if (length(args) >= 2) args[2] else "none"
emoset   <- if (length(args) >= 3) args[3] else "main"
stopifnot(lang %in% c("de_en", "de", "all"), controls %in% c("none", "activity"),
          emoset %in% c("main", "disgust"))

src <- "../../data"
dst <- "../../output"
file <- paste0("Austria-Panel-Users-Features-Tweet-", lang, ".csv")
emotions <- c("anger", "fear", if (emoset == "disgust") "disgust")
feature_types <- c(base = "baseline", var = "variability",
                   inst = "instability", inertia = "inertia")
outcomes <- c(news = "shares_news", misinfo = "shares_untrustworthy",
              partisan = "shares_biased")

features <- read_csv(file.path(src, file),
                     col_types = cols(author_id = col_character(),
                                      .default = col_double()))

features$shares_news <- as.integer(features$n_news_shared > 0)
features$shares_untrustworthy <- as.integer(features$n_untrustworthy_shared > 0)
features$shares_biased <- as.integer(features$n_biased_shared > 0)
features$log_n_scored <- log(features$n_scored)

feature_cols <- as.vector(outer(emotions, feature_types, paste, sep = "_"))
complete_features <- features[complete.cases(features[, c(unname(outcomes), feature_cols)]), ]
cat(sprintf("lang = %s, controls = %s, emotions = %s: %d of %d users with complete features\n",
            lang, controls, paste(emotions, collapse = "+"), nrow(complete_features), nrow(features)))

df <- complete_features %>%
  mutate(across(all_of(c(feature_cols, "log_n_scored")), ~ as.numeric(scale(.))))

boot_stat <- function(data, indices, formula) {
  d <- data[indices, ]
  model <- glm(formula, data = d, family = binomial)
  return(coef(model))
}

R <- 1000 # number of bootstrap samples
n_cores <- 20
set.seed(636)

boot_objects <- list()
for (o in names(outcomes)) {
  cat(sprintf("Running bootstrap for %s models...\n", toupper(o)))
  for (f in names(feature_types)) {
    rhs <- paste(paste(emotions, feature_types[[f]], sep = "_"), collapse = " + ")
    if (controls == "activity") rhs <- paste(rhs, "+ log_n_scored")
    fml <- as.formula(paste(outcomes[[o]], "~", rhs))
    boot_objects[[paste("boot", o, f, sep = "_")]] <-
      boot(data = df, statistic = boot_stat, R = R,
           parallel = "multicore", ncpus = n_cores, formula = fml)
  }
}

dir.create(file.path(dst, "users"), recursive = TRUE, showWarnings = FALSE)
out <- file.path(dst, "users", paste0("user_models_tweet_boot_", lang,
                                 if (controls == "activity") "_activity" else "",
                                 if (emoset == "disgust") "_disgust" else "", ".rds"))
saveRDS(boot_objects, out)
cat("Bootstrap analysis completed:", out, "\n")
