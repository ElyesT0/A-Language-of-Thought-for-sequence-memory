# --- Packages ---
library(lme4)
library(MuMIn)         # for r.squaredGLMM
library(dplyr)
library(purrr)
library(tibble)
library(stringr)
library(ggplot2)
library(readr)

# --- Load data ---
data <- read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_EXT_only_REP.csv")

# Ensure grouping factor is a factor
if (!"participant_ID" %in% names(data)) stop("Column 'participant_ID' not found in data.")
data$participant_ID <- as.factor(data$participant_ID)

# --- Predictors to compare (exact names from your script) ---
predictors <- c(
  "LoT.Complexity",
  "Subjective.Complexity",
  "Shannon.Entropy",
  "Shannon.Entropy.Bigram",
  "Lempel.Ziv",
  "Change.Complexity",
  "Change.Complexity.Extended",
  "Algorithmic.Complexity",
  "Subsymetries",
  "Chunk.Complexity.Local",
  "Chunk.Complexity.Global"
)

# --- Controls ---
nsim_boot <- 1000    # number of bootstrap replicates for CIs
set.seed(123)        # reproducibility for bootstraps

# --- Helper: fit one model, return tidy results with bootstrap CIs for fixed effects ---
fit_one_model <- function(pred, dat, nsim = 1000) {
  fml <- as.formula(paste0("distance_dl ~ `", pred, "` + (1 | participant_ID)"))
  model <- lmer(fml, data = dat, REML = FALSE)
  
  # Fixed effects table
  smry <- coef(summary(model))
  fe_df <- smry %>%
    as.data.frame() %>%
    rownames_to_column("term") %>%
    rename(estimate = Estimate,
           std_error = `Std. Error`,
           t_value = `t value`)
  
  # Bootstrap CIs for all fixed effects (beta_... includes intercept + terms)
  ci <- suppressMessages(
    confint(model, method = "boot", nsim = nsim, parm = "beta_")
  )
  ci_df <- as.data.frame(ci) %>%
    rownames_to_column("term") %>%
    rename(ci_low = `2.5 %`, ci_high = `97.5 %`)
  
  # Join
  out <- fe_df %>%
    left_join(ci_df, by = "term") %>%
    mutate(model_name = pred,
           AIC = AIC(model),
           BIC = BIC(model)) %>%
    relocate(model_name, .before = term)
  
  # R2 (marginal/conditional)
  r2 <- suppressWarnings(r.squaredGLMM(model))
  # r.squaredGLMM returns a matrix with 2 columns (R2m, R2c)
  out$R2_marginal <- as.numeric(r2[1])
  out$R2_conditional <- as.numeric(r2[2])
  
  # Also add (approx) two-sided p-values from t (as you did)
  out$p_value_approx <- 2 * (1 - pnorm(abs(out$t_value)))
  
  # Return model object too if you want to inspect later
  attr(out, "model") <- model
  out
}

# --- Fit all models and bind results ---
all_results <- predictors %>%
  set_names() %>%
  map(~ fit_one_model(.x, data, nsim = nsim_boot)) %>%
  list_rbind()

# Preview
print(all_results %>% select(model_name, term, estimate, std_error, t_value, p_value_approx, ci_low, ci_high, AIC, BIC, R2_marginal, R2_conditional) %>% head(20))

# --- Plot: fixed-effect coefficients with 95% bootstrap CIs (excluding intercept) ---
coef_plot_df <- all_results %>%
  filter(term != "(Intercept)")

ggplot(coef_plot_df,
       aes(x = reorder(model_name, estimate),
           y = estimate)) +
  geom_pointrange(aes(ymin = ci_low, ymax = ci_high)) +
  coord_flip() +
  labs(title = "Fixed-effect estimates with 95% bootstrap CIs",
       x = "Predictor (model)",
       y = "Coefficient estimate on distance_dl") +
  theme_minimal(base_size = 13)

# --- Table: model-level comparison (AIC/BIC/R2), one row per model ---
model_level <- coef_plot_df %>%
  group_by(model_name) %>%
  summarise(
    AIC = first(AIC),
    BIC = first(BIC),
    R2_marginal = first(R2_marginal),
    R2_conditional = first(R2_conditional),
    .groups = "drop"
  ) %>%
  arrange(AIC)

print(model_level)

# ======================================================================
# OPTIONAL: Bootstrap CIs for group means (e.g., by LoT.Complexity) for plotting error bars
# ======================================================================
# This is useful if you want to visualize the average distance_dl per level of a (categorical) complexity,
# with bootstrap 95% CIs. Works best when the predictor is discrete/factor.

bootstrap_group_means <- function(dat, group_var, value_var = "distance_dl", R = 1000) {
  stopifnot(group_var %in% names(dat), value_var %in% names(dat))
  g <- dat[[group_var]]
  v <- dat[[value_var]]
  
  # Ensure factor for grouping (keeps level order if already factor)
  g <- as.factor(g)
  
  # For each level, bootstrap the mean
  levels_g <- levels(g)
  
  res <- map_dfr(levels_g, function(lv) {
    vals <- v[g == lv]
    # Non-parametric bootstrap: resample indices
    means <- replicate(R, mean(sample(vals, replace = TRUE), na.rm = TRUE))
    tibble(
      !!group_var := lv,
      mean = mean(vals, na.rm = TRUE),
      ci_low = quantile(means, 0.025, na.rm = TRUE),
      ci_high = quantile(means, 0.975, na.rm = TRUE)
    )
  })
  
  res
}

# Example usage for LoT.Complexity (only if it is discrete/categorical):
if ("LoT.Complexity" %in% names(data)) {
  lot_group_summary <- bootstrap_group_means(data, "LoT.Complexity", "distance_dl", R = 1000)
  
  ggplot(lot_group_summary, aes(x = as.factor(LoT.Complexity), y = mean)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2) +
    labs(title = "Mean distance_dl by LoT.Complexity with 95% bootstrap CIs",
         x = "LoT.Complexity",
         y = "Mean distance_dl") +
    theme_minimal(base_size = 13)
}

# --- End ---
