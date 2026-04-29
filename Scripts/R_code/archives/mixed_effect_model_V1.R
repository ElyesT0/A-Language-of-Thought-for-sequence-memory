# Specify the file path for the log file
log_file <- "/Users/et/Documents/UNICOG/6-publication/1-sequence_learning_cognition/article_memocrush/Results-Resources/reports_models_stats/mixed_effect_model.txt"

# Open the connection to the log file
sink(log_file)

# Include a timestamp or header for the log
cat("============================================\n")
cat("             ANALYSIS LOG FILE              \n")
cat("============================================\n")
cat("Log start time: ", Sys.time(), "\n")
cat("============================================\n\n")

# Include a timestamp or header for the log (optional)
cat("Log start time: ", Sys.time(), "\n\n")

library(lme4)
library(MuMIn)  # Load MuMIn package for R-squared calculation

# Load the data
data_BASE <- read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_BASE_only.csv")
data_ext <- read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_EXT_only.csv")
data_complete <- read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset.csv")

# Ensure column names are correctly referenced
colnames(data_BASE) <- gsub(" ", "_", colnames(data_BASE))
colnames(data_ext) <- gsub(" ", "_", colnames(data_ext))
colnames(data_complete) <- gsub(" ", "_", colnames(data_complete))

cat("*********************************************\n")
cat("             LOT COMPLEXITY            \n")
cat("********************************************\n")

# Function to print a section header
print_section_header <- function(title) {
  cat("\n--------------------------------------------\n")
  cat("             ", title, "\n")
  cat("--------------------------------------------\n\n")
}

# Function to calculate and print ICC and R-squared values
calculate_icc_and_r2 <- function(model) {
  var_random <- as.numeric(VarCorr(model)$participant_ID[1])
  var_residual <- attr(VarCorr(model), "sc")^2
  icc <- var_random / (var_random + var_residual)
  
  r2_marginal <- r.squaredGLMM(model)[1]
  r2_conditional <- r.squaredGLMM(model)[2]
  
  cat("Intra-class Correlation Coefficient (ICC): ", round(icc, 4), "\n")
  cat("This indicates that approximately ", round(icc * 100, 2), "% of the variance in performance is explained by individual differences.\n")
  
  cat("Marginal R-squared (variance explained by LoT complexity): ", round(r2_marginal, 4), "\n")
  cat("Conditional R-squared (variance explained by both LoT complexity and individual differences): ", round(r2_conditional, 4), "\n\n")
}

# Fit the mixed-effects model for data_complete
print_section_header("DATA COMPLETE: Mixed-Effects Model")
cat("Fitting model...\n\n")

# Fit the mixed-effects model and calculate ICC and R-squared for data_complete
mixed_model_complete <- lmer(distance_dl ~ LoT.Complexity + (1 | participant_ID), data = data_complete)
cat("Model summary:\n")
print(summary(mixed_model_complete))
calculate_icc_and_r2(mixed_model_complete)

# Fit the mixed-effects model for data_BASE
print_section_header("DATA BASE: Mixed-Effects Model")
cat("Fitting model...\n\n")
mixed_model_base <- lmer(distance_dl ~ LoT.Complexity + (1 | participant_ID), data = data_BASE)
cat("Model summary:\n")
print(summary(mixed_model_base))
calculate_icc_and_r2(mixed_model_base)

# Fit the mixed-effects model for data_ext
print_section_header("DATA EXT: Mixed-Effects Model")
cat("Fitting model...\n\n")
mixed_model_ext <- lmer(distance_dl ~ LoT.Complexity + (1 | participant_ID), data = data_ext)
cat("Model summary:\n")
print(summary(mixed_model_ext))
calculate_icc_and_r2(mixed_model_ext)

cat("*********************************************\n")
cat("             Lempel_Ziv            \n")
cat("********************************************\n")

# Fit the mixed-effects model for data_complete
print_section_header("DATA COMPLETE: Mixed-Effects Model")
cat("Fitting model...\n\n")

# Fit the mixed-effects model and calculate ICC and R-squared for data_complete
mixed_model_complete <- lmer(distance_dl ~ Lempel_Ziv  + (1 | participant_ID), data = data_complete)
cat("Model summary:\n")
print(summary(mixed_model_complete))
calculate_icc_and_r2(mixed_model_complete)

# Fit the mixed-effects model for data_BASE
print_section_header("DATA BASE: Mixed-Effects Model")
cat("Fitting model...\n\n")
mixed_model_base <- lmer(distance_dl ~ Lempel_Ziv  + (1 | participant_ID), data = data_BASE)
cat("Model summary:\n")
print(summary(mixed_model_base))
calculate_icc_and_r2(mixed_model_base)

# Fit the mixed-effects model for data_ext
print_section_header("DATA EXT: Mixed-Effects Model")
cat("Fitting model...\n\n")
mixed_model_ext <- lmer(distance_dl ~ Lempel_Ziv  + (1 | participant_ID), data = data_ext)
cat("Model summary:\n")
print(summary(mixed_model_ext))
calculate_icc_and_r2(mixed_model_ext)

cat("*********************************************\n")
cat("             Algorithmic.Complexity            \n")
cat("********************************************\n")

# Fit the mixed-effects model for data_complete
print_section_header("DATA COMPLETE: Mixed-Effects Model")
cat("Fitting model...\n\n")

# Fit the mixed-effects model and calculate ICC and R-squared for data_complete
mixed_model_complete <- lmer(distance_dl ~ Algorithmic.Complexity  + (1 | participant_ID), data = data_complete)
cat("Model summary:\n")
print(summary(mixed_model_complete))
calculate_icc_and_r2(mixed_model_complete)

# Fit the mixed-effects model for data_BASE
print_section_header("DATA BASE: Mixed-Effects Model")
cat("Fitting model...\n\n")
mixed_model_base <- lmer(distance_dl ~ Algorithmic.Complexity  + (1 | participant_ID), data = data_BASE)
cat("Model summary:\n")
print(summary(mixed_model_base))
calculate_icc_and_r2(mixed_model_base)

# Fit the mixed-effects model for data_ext
print_section_header("DATA EXT: Mixed-Effects Model")
cat("Fitting model...\n\n")
mixed_model_ext <- lmer(distance_dl ~ Algorithmic.Complexity  + (1 | participant_ID), data = data_ext)
cat("Model summary:\n")
print(summary(mixed_model_ext))
calculate_icc_and_r2(mixed_model_ext)

# Close the connection to the log file
sink()

# Optionally, open the log file to view
file.show(log_file)
