# Task:   Bootstrapped LM/LMER models: does the trustworthiness of shared news predict emotions
#         after sharing?
# Input:  data/Austria-Panel-Windows-<size>-10min-<lang>.csv.gz
# Output: scripts/analysis/windows_models/windows_10_<size>_<lang>/ (models, bootstrap draws, coefficients)
# Run:    Rscript 8-boot_window_models.R <size> [lang] [emotions]   # size: 1h or 24h; defaults: de_en main
rm(list = ls())
library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(boot)
library(parallel)
args <- commandArgs(trailingOnly = TRUE)

assignInNamespace(".check_ncores", function(nc) {}, ns = "parallel")

set.seed(636)
n_boot <- 1000
n_cores <- 15
window_size <- args[1]  # "1h" or "24h"
lang <- if (length(args) >= 2) args[2] else "de_en"
emoset <- if (length(args) >= 3) args[3] else "main"
stopifnot(window_size %in% c("1h", "24h"), lang %in% c("de_en", "de", "all"),
          emoset %in% c("main", "disgust"))
dst <- file.path("windows_models", paste0("windows_10_", window_size, "_", lang))
dir.create(dst, showWarnings = FALSE, recursive = TRUE)

src <- "../../data"
# each emotion is modeled separately, so the disgust models do not affect anger and fear
emotions <- if (emoset == "disgust") "disgust" else c("anger", "fear")
emo_cols <- as.vector(outer(c("mean"), as.vector(outer(emotions, c("before", "after"), paste, sep = "_")),
                            paste, sep = "_"))

window_data <- read_csv(
  file.path(src, paste0("Austria-Panel-Windows-", window_size, "-10min-", lang, ".csv.gz")),
  col_select = all_of(c("event_id", "author_id", "event_score", emo_cols)),
  col_types = cols(event_id = col_character(), author_id = col_character(),
                   .default = col_double())
)

cat(sprintf("[%s] Preparing data...\n", Sys.time()))
window_data <- window_data %>%
  mutate(untrustworthy = (100 - event_score) / 100)

cat(sprintf("[%s] Fitting models for\n%d obs. from %d users...\n", Sys.time(),
            nrow(window_data), n_distinct(window_data$author_id)))

# model specifications per emotion:
# before: untrustworthy ~ emotion before; after: emotion after ~ untrustworthy;
# change: emotion after ~ untrustworthy + emotion before; each as lm and lmer
specs <- function(e) {
  b <- paste0("mean_", e, "_before"); a <- paste0("mean_", e, "_after")
  list(
    before = list(fml = paste("untrustworthy ~", b), param = b),
    after  = list(fml = paste(a, "~ untrustworthy"), param = "untrustworthy"),
    change = list(fml = paste(a, "~ untrustworthy +", b), param = "untrustworthy")
  )
}

# Bootstrap function for re-sampling data at the user level
boot_coef <- function(data, model_formula, param_name, model_type = "lm") {
  # Resample users (cluster bootstrap)
  users <- unique(data$author_id)
  boot_users <- data.frame(author_id = sample(users, replace = TRUE),
                           boot_id = seq_along(users))
  # Join to duplicate rows for users sampled multiple times
  boot_data <- boot_users %>%
    left_join(data, by = "author_id", relationship = "many-to-many")

  if (model_type == "lm") {
    boot_model <- lm(model_formula, data = boot_data)
    return(list(fixed_effect = coef(boot_model)[param_name], random_effects = NULL))
  } else {
    boot_model <- lmer(model_formula, data = boot_data,
                       control = lmerControl(calc.derivs = FALSE))
    re <- ranef(boot_model)$author_id
    random_effects <- fixef(boot_model)[1] + re[, 1]
    names(random_effects) <- rownames(re)
    return(list(fixed_effect = fixef(boot_model)[param_name],
                random_effects = random_effects))
  }
}

# Helper function to extract fixed effect CIs
extract_coef_ci <- function(boot_list, original_model, param_name,
                            model_type, window, emotion) {
  fixed_boot <- sapply(boot_list, function(x) x$fixed_effect)
  ci <- quantile(fixed_boot, c(0.025, 0.975), na.rm = TRUE)
  tidy(original_model) %>%
    filter(term == param_name) %>%
    mutate(ci_lower = ci[1], ci_upper = ci[2], model = model_type,
           window = window, emotion = emotion, n_boot = length(fixed_boot),
           boot_mean = mean(fixed_boot, na.rm = TRUE),
           boot_sd = sd(fixed_boot, na.rm = TRUE))
}

coefs <- list()
for (e in emotions) {
  cat(sprintf("[%s] %s models...\n", Sys.time(), e))
  for (w in names(specs(e))) {
    s <- specs(e)[[w]]
    for (type in c("lm", "lmer")) {
      fml <- as.formula(if (type == "lm") s$fml else paste(s$fml, "+ (1 | author_id)"))
      m <- if (type == "lm") lm(fml, data = window_data) else lmer(fml, data = window_data)
      saveRDS(m, file = file.path(dst, paste0(type, "_", e, "_", w, ".rds")))

      b <- mclapply(1:n_boot, function(i) {
        boot_coef(window_data, fml, s$param, model_type = type)
      }, mc.cores = n_cores)
      saveRDS(b, file = file.path(dst, paste0("boot_", type, "_", e, "_", w, ".rds")))

      coefs[[paste(e, w, type)]] <- extract_coef_ci(b, m, s$param, type, w, e)
    }
  }
}

# Save coefficient table
coef_combined <- bind_rows(coefs)
coef_file <- if (emoset == "disgust") "coefs_window_models_disgust.csv" else "coefs_window_models.csv"
write_csv(coef_combined, file.path(dst, coef_file))
cat(sprintf("[%s] Coefficient table saved to: %s\n", Sys.time(),
            file.path(dst, coef_file)))
