# ===== SCRIPT OVERVIEW =====
# Task:   Fit and cluster-bootstrap LMER models testing whether exposure to
#         untrustworthy news predicts subsequent emotion change.
#         Models fitted (lm and lmer with random intercepts per author_id):
#           anger_after ~ untrustworthy [+ anger_before]  (and mirrored for fear)
# Input:  data/Austria-Panel-Windows-{1h|24h}-10min.csv.gz
# Output: ./windows_10_{1h|24h}/*.csv — bootstrapped fixed-effect CIs
# Usage:  Rscript 8-boot_window_models.R <1h|24h>

library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(boot)
library(parallel)

args <- commandArgs(trailingOnly = TRUE)

assignInNamespace(".check_ncores", function(nc) {}, ns = "parallel")

set.seed(636)
n_boot <- 10
n_cores <- 10
window_size <- args[1]  # "1h" or "24h"
dst <- file.path("windows_models", paste0("windows_10_", window_size))
dir.create(dst, showWarnings = FALSE, recursive = TRUE)

src <- "../../data"

window_data <- read_csv(file.path(src, paste0("Austria-Panel-Windows-", window_size, "-10min.csv.gz")),
  col_select = c(
    event_id, 
    author_id, 
    event_score, 
    mean_anger_before,
    mean_fear_before,
    mean_anger_after,
    mean_fear_after
  ),
  col_types = cols(
    event_id = col_character(),
    author_id = col_character(),
    event_score = col_double(),
    mean_anger_before = col_double(),
    mean_fear_before = col_double(),
    mean_anger_after = col_double(),
    mean_fear_after = col_double()
  )
)

cat(sprintf("[%s] Preparing data...\n", Sys.time()))
window_data <- window_data %>%
  mutate(
    untrustworthy = (100 - event_score) / 100,
  ) 

cat(sprintf("[%s] Fitting LMER models for\n%d obs. from %d users...\n", Sys.time(),
            nrow(window_data),
            n_distinct(window_data$author_id)))

# fitting simple linear models 
anger_m1 <- lm(untrustworthy ~ mean_anger_before, data = window_data)
anger_m2 <- lmer(untrustworthy ~ mean_anger_before + (1 | author_id), data = window_data)
anger_m3 <- lm(mean_anger_after ~ untrustworthy, data = window_data)
anger_m4 <- lmer(mean_anger_after ~ untrustworthy + (1 | author_id), data = window_data)
anger_m5 <- lm(mean_anger_after ~ untrustworthy + mean_anger_before, data = window_data)
anger_m6 <- lmer(mean_anger_after ~ untrustworthy + mean_anger_before + (1 | author_id), data = window_data)

fear_m1 <- lm(untrustworthy ~ mean_fear_before, data = window_data)
fear_m2 <- lmer(untrustworthy ~ mean_fear_before + (1 | author_id), data = window_data)
fear_m3 <- lm(mean_fear_after ~ untrustworthy, data = window_data)
fear_m4 <- lmer(mean_fear_after ~ untrustworthy + (1 | author_id), data = window_data)
fear_m5 <- lm(mean_fear_after ~ untrustworthy + mean_fear_before, data = window_data)
fear_m6 <- lmer(mean_fear_after ~ untrustworthy + mean_fear_before + (1 | author_id), data = window_data)

saveRDS(anger_m1, file = file.path(dst, "lm_anger_before.rds"))
saveRDS(anger_m2, file = file.path(dst, "lmer_anger_before.rds"))
saveRDS(anger_m3, file = file.path(dst, "lm_anger_after.rds"))
saveRDS(anger_m4, file = file.path(dst, "lmer_anger_after.rds"))
saveRDS(anger_m5, file = file.path(dst, "lm_anger_change.rds"))
saveRDS(anger_m6, file = file.path(dst, "lmer_anger_change.rds"))

saveRDS(fear_m1, file = file.path(dst, "lm_fear_before.rds"))
saveRDS(fear_m2, file = file.path(dst, "lmer_fear_before.rds"))
saveRDS(fear_m3, file = file.path(dst, "lm_fear_after.rds"))
saveRDS(fear_m4, file = file.path(dst, "lmer_fear_after.rds"))
saveRDS(fear_m5, file = file.path(dst, "lm_fear_change.rds"))
saveRDS(fear_m6, file = file.path(dst, "lmer_fear_change.rds"))

cat(sprintf("[%s] Bootstrapping confidence intervals...\n", Sys.time()))

# Bootstrap function for re-sampling data at the user level
boot_coef <- function(data, model_formula, param_name, model_type = "lm") {
  
  # Resample users (cluster bootstrap)
  users <- unique(data$author_id)
  boot_users <- data.frame(
    author_id = sample(users, replace = TRUE),
    boot_id = seq_along(users)
  )
  
  # Join to duplicate rows for users sampled multiple times
  boot_data <- boot_users %>%
    left_join(data, by = "author_id", relationship = "many-to-many")
  
  # Re-fit model based on type
  if (model_type == "lm") {
    boot_model <- lm(model_formula, data = boot_data)
    fixed_coef <- coef(boot_model)[param_name]
    
    return(list(
      fixed_effect = fixed_coef,
      random_effects = NULL
    ))
    
  } else if (model_type == "lmer") {
    boot_model <- lmer(model_formula, data = boot_data,
                       control = lmerControl(calc.derivs = FALSE))
    fixed_coef <- fixef(boot_model)[param_name]
    
    # Extract random effects
    re <- ranef(boot_model)$author_id
    fixed_int <- fixef(boot_model)[1]
    random_effects <- fixed_int + re[,1]
    names(random_effects) <- rownames(re)
    
    return(list(
      fixed_effect = fixed_coef,
      random_effects = random_effects
    ))
  }
}

# Bootstrap confidence intervals for each model
cat("  Bootstrapping anger models...\n")
anger_m1_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            untrustworthy ~ mean_anger_before,
            "mean_anger_before",
            model_type = "lm")
}, mc.cores = n_cores)

saveRDS(anger_m1_boot, file = file.path(dst, "boot_lm_anger_before.rds"))

anger_m2_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            untrustworthy ~ mean_anger_before + (1 | author_id),
            "mean_anger_before",
            model_type = "lmer")
}, mc.cores = n_cores)

saveRDS(anger_m2_boot, file = file.path(dst, "boot_lmer_anger_before.rds"))

anger_m3_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_anger_after ~ untrustworthy,
            "untrustworthy",
            model_type = "lm")
}, mc.cores = n_cores)
saveRDS(anger_m3_boot, file = file.path(dst, "boot_lm_anger_after.rds"))

anger_m4_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_anger_after ~ untrustworthy + (1 | author_id),
            "untrustworthy",
            model_type = "lmer")
}, mc.cores = n_cores)
saveRDS(anger_m4_boot, file = file.path(dst, "boot_lmer_anger_after.rds"))

anger_m5_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_anger_after ~ untrustworthy + mean_anger_before,
            "untrustworthy",
            model_type = "lm")
}, mc.cores = n_cores)
saveRDS(anger_m5_boot, file = file.path(dst, "boot_lm_anger_change.rds"))

anger_m6_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_anger_after ~ untrustworthy + mean_anger_before + (1 | author_id),
            "untrustworthy",
            model_type = "lmer")
}, mc.cores = n_cores)
saveRDS(anger_m6_boot, file = file.path(dst, "boot_lmer_anger_change.rds"))

cat("  Bootstrapping fear models...\n")
fear_m1_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            untrustworthy ~ mean_fear_before,
            "mean_fear_before",
            model_type = "lm")
}, mc.cores = n_cores)
saveRDS(fear_m1_boot, file = file.path(dst, "boot_lm_fear_before.rds"))

fear_m2_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            untrustworthy ~ mean_fear_before + (1 | author_id),
            "mean_fear_before",
            model_type = "lmer")
}, mc.cores = n_cores)
saveRDS(fear_m2_boot, file = file.path(dst, "boot_lmer_fear_before.rds"))

fear_m3_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_fear_after ~ untrustworthy,
            "untrustworthy",
            model_type = "lm")
}, mc.cores = n_cores)
saveRDS(fear_m3_boot, file = file.path(dst, "boot_lm_fear_after.rds"))

fear_m4_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_fear_after ~ untrustworthy + (1 | author_id),
            "untrustworthy",
            model_type = "lmer")
}, mc.cores = n_cores)
saveRDS(fear_m4_boot, file = file.path(dst, "boot_lmer_fear_after.rds"))

fear_m5_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_fear_after ~ untrustworthy + mean_fear_before,
            "untrustworthy",
            model_type = "lm")
}, mc.cores = n_cores)
saveRDS(fear_m5_boot, file = file.path(dst, "boot_lm_fear_change.rds"))

fear_m6_boot <- mclapply(1:n_boot, function(i) {
  boot_coef(window_data, 
            mean_fear_after ~ untrustworthy + mean_fear_before + (1 | author_id),
            "untrustworthy",
            model_type = "lmer")
}, mc.cores = n_cores)
saveRDS(fear_m6_boot, file = file.path(dst, "boot_lmer_fear_change.rds"))
# ===== EXTRACT BOOTSTRAP CIs FOR COEFFICIENTS =====
cat(sprintf("[%s] Extracting coefficient confidence intervals...\n", Sys.time()))

# Helper function to extract fixed effect CIs
extract_coef_ci <- function(boot_list, original_model, param_name, 
                            model_type, window, emotion) {
  
  # Extract fixed effects from bootstrap
  fixed_boot <- sapply(boot_list, function(x) x$fixed_effect)
  
  # Calculate percentile CIs
  ci <- quantile(fixed_boot, c(0.025, 0.975), na.rm = TRUE)
  
  # Get original coefficient
  coef_df <- tidy(original_model) %>%
    filter(term == param_name) %>%
    mutate(
      ci_lower = ci[1],
      ci_upper = ci[2],
      model = model_type,
      window = window,
      emotion = emotion,
      n_boot = length(fixed_boot),
      boot_mean = mean(fixed_boot, na.rm = TRUE),
      boot_sd = sd(fixed_boot, na.rm = TRUE)
    )
  
  return(coef_df)
}

# Extract for all models
coef_anger_m1 <- extract_coef_ci(
  anger_m1_boot, anger_m1, "mean_anger_before", 
  "lm", "before", "anger"
)

coef_anger_m2 <- extract_coef_ci(
  anger_m2_boot, anger_m2, "mean_anger_before", 
  "lmer", "before", "anger"
)

coef_anger_m3 <- extract_coef_ci(
  anger_m3_boot, anger_m3, "untrustworthy", 
  "lm", "after", "anger"
)

coef_anger_m4 <- extract_coef_ci(
  anger_m4_boot, anger_m4, "untrustworthy", 
  "lmer", "after", "anger"
)

coef_anger_m5 <- extract_coef_ci(
  anger_m5_boot, anger_m5, "untrustworthy", 
  "lm", "change", "anger"
)

coef_anger_m6 <- extract_coef_ci(
  anger_m6_boot, anger_m6, "untrustworthy", 
  "lmer", "change", "anger"
)

coef_fear_m1 <- extract_coef_ci(
  fear_m1_boot, fear_m1, "mean_fear_before", 
  "lm", "before", "fear"
)

coef_fear_m2 <- extract_coef_ci(
  fear_m2_boot, fear_m2, "mean_fear_before", 
  "lmer", "before", "fear"
)

coef_fear_m3 <- extract_coef_ci(
  fear_m3_boot, fear_m3, "untrustworthy", 
  "lm", "after", "fear"
)

coef_fear_m4 <- extract_coef_ci(
  fear_m4_boot, fear_m4, "untrustworthy", 
  "lmer", "after", "fear"
)

coef_fear_m5 <- extract_coef_ci(
  fear_m5_boot, fear_m5, "untrustworthy", 
  "lm", "change", "fear"
)

coef_fear_m6 <- extract_coef_ci(
  fear_m6_boot, fear_m6, "untrustworthy", 
  "lmer", "change", "fear"
)

# Combine all coefficients
coef_combined <- bind_rows(
  coef_anger_m1, coef_anger_m2, coef_anger_m3, 
  coef_anger_m4, coef_anger_m5, coef_anger_m6,
  coef_fear_m1, coef_fear_m2, coef_fear_m3, 
  coef_fear_m4, coef_fear_m5, coef_fear_m6
)

# Save coefficient table
write_csv(coef_combined, file.path(dst, "coefs_window_models.csv"))

cat(sprintf("[%s] Coefficient table saved to: %s\n", 
            Sys.time(), 
            file.path(dst, "coefs_window_models.csv")))