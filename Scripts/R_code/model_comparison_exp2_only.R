library(lme4)

# Load data
data <- read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_EXT_only_REP.csv")

# Boxplots for different complexity measures
boxplot(distance_dl ~ LoT.Complexity, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Subjective.Complexity, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Shannon.Entropy, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Shannon.Entropy.Bigram, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Lempel.Ziv, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Change.Complexity, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Change.Complexity.Extended, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Algorithmic.Complexity, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Subsymetries, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Chunk.Complexity.Local, col=c("white","lightgray"), data)
boxplot(distance_dl ~ Chunk.Complexity.Global, col=c("white","lightgray"), data)

# Function to fit model and print statistics
fit_model <- function(formula, data, model_name) {
  model <- lmer(formula, data = data)
  AIC_value <- AIC(model)
  cat(sprintf("\n%s:\n", model_name))
  cat(sprintf("AIC: %f\n", AIC_value))
  
  # Extract model summary
  model_summary <- coef(summary(model))
  
  # Print statistics for fixed effects
  print(data.frame(
    Estimate = model_summary[, "Estimate"],
    SE = model_summary[, "Std. Error"],
    t_value = model_summary[, "t value"],
    p_value = 2 * (1 - pnorm(abs(model_summary[, "t value"]))) # Approximate p-values
  ), row.names = TRUE)
}

# Fit models
fit_model(distance_dl ~ LoT.Complexity + (1 | participant_ID), data, "LoT Complexity")
fit_model(distance_dl ~ Subjective.Complexity + (1 | participant_ID), data, "Subjective Complexity")
fit_model(distance_dl ~ Shannon.Entropy + (1 | participant_ID), data, "Shannon Entropy")
fit_model(distance_dl ~ Shannon.Entropy.Bigram + (1 | participant_ID), data, "Shannon Entropy Bigram")
fit_model(distance_dl ~ Lempel.Ziv + (1 | participant_ID), data, "Lempel Ziv Complexity")
fit_model(distance_dl ~ Change.Complexity + (1 | participant_ID), data, "Change Complexity")
fit_model(distance_dl ~ Change.Complexity.Extended + (1 | participant_ID), data, "Change Complexity Extended")
fit_model(distance_dl ~ Algorithmic.Complexity + (1 | participant_ID), data, "Algorithmic Complexity")
fit_model(distance_dl ~ Subsymetries + (1 | participant_ID), data, "Subsymetries")
fit_model(distance_dl ~ Chunk.Complexity.Local + (1 | participant_ID), data, "Chunk Complexity Local")
fit_model(distance_dl ~ Chunk.Complexity.Global + (1 | participant_ID), data, "Chunk Complexity Global")
