# Load required libraries
library(lme4)
library(performance)

# Specify the file path for the log file
log_file <- "/Users/elyestabbane/Documents/UNICOG/6-publication/1-sequence_learning_cognition/article_memocrush/Results-Resources/reports_models_stats/mixed_effect_model.txt"

# Function to print a section header
print_section_header <- function(title) {
  cat("\n--------------------------------------------\n")
  cat("             ", title, "\n")
  cat("--------------------------------------------\n\n")
}

calculate_metrics <- function(model) {
  var_random <- as.numeric(VarCorr(model)$participant_ID[1])
  var_residual <- attr(VarCorr(model), "sc")^2
  icc <- var_random / (var_random + var_residual)
  
  r2_values <- r2(model)  # Calculate R-squared using performance package
  aic_value <- AIC(model) # Calculate AIC
  
  cat("Intra-class Correlation Coefficient (ICC): ", round(icc, 4), "\n")
  cat("This indicates that approximately ", round(icc * 100, 2), "% of the variance in performance is explained by individual differences.\n")
  
  cat("Marginal R-squared (variance explained by fixed effect): ", round(r2_values$R2_marginal, 4), "\n")
  cat("Conditional R-squared (variance explained by both fixed effect and individual differences): ", round(r2_values$R2_conditional, 4), "\n")
  
  cat("Akaike Information Criterion (AIC): ", round(aic_value, 4), "\n\n")
}

# General function to fit and log mixed-effects models
fit_and_log_models <- function(data_list, complexities, response_variable) {
  # Open the connection to the log file
  sink(log_file)
  
  # Include a timestamp or header for the log
  cat("============================================\n")
  cat("             ANALYSIS LOG FILE              \n")
  cat("============================================\n")
  cat("Log start time: ", Sys.time(), "\n")
  cat("============================================\n\n")
  
  # Loop through each complexity type
  for (complexity in complexities) {
    cat("*********************************************\n")
    cat("             ", complexity, "            \n")
    cat("********************************************\n")
    
    for (data_name in names(data_list)) {
      data <- data_list[[data_name]]
      
      print_section_header(paste("DATA", toupper(data_name), ": Mixed-Effects Model"))
      cat("Fitting model...\n\n")
      
      # Create the formula
      formula <- as.formula(paste(response_variable, "~", complexity, "+ (1 | participant_ID)"))
      
      # Fit the model
      mixed_model <- lmer(formula, data = data)
      
      # Print the summary and calculate ICC and R-squared
      cat("Model summary:\n")
      print(summary(mixed_model))
      calculate_metrics(mixed_model)

    }
  }
  
  # Close the connection to the log file
  sink()
  
  # Optionally, open the log file to view
  file.show(log_file)
}

# Load the data
data_BASE <- read.csv("/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_BASE_only.csv")
data_ext <- read.csv("/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_EXT_only.csv")
data_complete <- read.csv("/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset.csv")

# Ensure column names are correctly referenced
colnames(data_BASE) <- gsub(" ", "_", colnames(data_BASE))
colnames(data_ext) <- gsub(" ", "_", colnames(data_ext))
colnames(data_complete) <- gsub(" ", "_", colnames(data_complete))

# Define the list of datasets
data_list <- list(
  base = data_BASE,
  ext = data_ext,
  complete = data_complete
)

# Define the list of complexities to iterate over
complexities <- c("LoT.Complexity", "Shannon.Entropy", "Lempel_Ziv","Change.Complexity","Algorithmic.Complexity","Subsymetries","Chunk.Complexity" )

# Call the function with the response variable
fit_and_log_models(data_list, complexities, "distance_dl")
