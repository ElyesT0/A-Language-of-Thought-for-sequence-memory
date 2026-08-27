library(lme4)
library(lmerTest)
library(car)

# ── Script path ───────────────────────────────────────────────────────────────

script_args <- commandArgs(trailingOnly = FALSE)
script_file <- sub("--file=", "", script_args[grep("--file=", script_args)])
if (length(script_file) == 0) script_file <- "(interactive session)"
script_file <- normalizePath(script_file, mustWork = FALSE)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      <- "/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush"
data_path <- file.path(ROOT,
  "Data/processed/both/two_experiment_datasets_2024-08_22.csv")
out_dir_exp1 <- file.path(ROOT, "Figures/review_response/geometry_analysis/exp1")
out_dir_exp2 <- file.path(ROOT, "Figures/review_response/geometry_analysis/exp2")

# ── Load and prepare data ─────────────────────────────────────────────────────

df <- read.csv(data_path, stringsAsFactors = FALSE)
df$starting_vertex <- factor(as.integer(sub("\\[([0-9]+),.*", "\\1", df$seq)))
df$seq_name        <- factor(df$seq_name)
df                 <- df[df$seq_name != "Training", ]

# Binary outcome: 1 = error, 0 = correct
df$fail <- as.integer(df$performance == "fail")

# Total response time: sum of inter-click intervals (ms), from first to last press
parse_sum <- function(s) {
  if (is.na(s) || nchar(trimws(s)) == 0) return(NA_real_)
  vals <- suppressWarnings(
    as.integer(strsplit(gsub("[\\[\\] ]", "", s, perl = TRUE), ",")[[1]])
  )
  if (all(is.na(vals))) return(NA_real_)
  sum(vals, na.rm = TRUE)
}
df$response_time <- sapply(df$interclick_time, parse_sum)

# ── Sequence lists ────────────────────────────────────────────────────────────

seq_names_exp1 <- c(
  "Repetition-2",        "control Repetition-2",
  "Repetition-3",        "control Repetition-3",
  "Repetition-4",        "control Repetition-4",
  "Repetition-Nested",   "control NoLocal nested",
  "control NoGlobal nested"
)

seq_names_exp2 <- c(
  "play 4 tokens",       "control play 4 tokens",
  "sub-programs 1",      "control sub-programs 1",
  "sub-programs 2",      "control sub-programs 2",
  "index i",             "control index i",
  "play",                "control play",
  "Insertion",           "Suppression",
  "Mirror-Rep",          "control Mirror-Rep",
  "Mirror-NoRep",        "control Mirror-NoRep"
)

# ── Logger ────────────────────────────────────────────────────────────────────

make_logger <- function() {
  lines <- character(0)
  list(
    log = function(msg = "") {
      cat(msg, "\n", sep = "")
      lines <<- c(lines, msg)
    },
    save = function(path) {
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      writeLines(lines, path)
      cat("Report saved:", path, "\n")
    }
  )
}

# ── Omnibus model for one outcome ─────────────────────────────────────────────
#
# For continuous outcomes (distance_dl, response_time): lmer + lmerTest F-tests
#   (Satterthwaite denominator df).
# For binary outcome (fail): glmer, binomial family + car::Anova Wald chi-square.
#
# Reports the starting_vertex main effect — relevant for ruling out a starting-position confound.

run_omnibus <- function(data, seq_names, outcome, logger) {
  sub            <- data[data$seq_name %in% seq_names & !is.na(data[[outcome]]), ]
  sub$starting_vertex <- droplevels(sub$starting_vertex)

  is_binary <- outcome == "fail"
  formula_str <- sprintf(
    "%s ~ starting_vertex + (1 | participant_ID)", outcome
  )
  frm <- as.formula(formula_str)

  if (is_binary) {
    fit <- tryCatch(
      glmer(frm, data = sub, family = binomial,
            control = glmerControl(optimizer = "bobyqa",
                                   optCtrl = list(maxfun = 2e5))),
      error = function(e) {
        logger$log(sprintf("  Model failed: %s", e$message)); NULL
      }
    )
  } else {
    fit <- tryCatch(
      lmer(frm, data = sub, REML = FALSE),
      error = function(e) {
        logger$log(sprintf("  Model failed: %s", e$message)); NULL
      }
    )
  }
  if (is.null(fit)) return(invisible(NULL))

  if (is_binary) {
    # Wald chi-square from car::Anova (type III)
    an <- Anova(fit, type = 3)
    terms_of_interest <- c("starting_vertex")
    for (term in terms_of_interest) {
      if (!term %in% rownames(an)) next
      chi2 <- an[term, "Chisq"]
      df_t <- an[term, "Df"]
      p    <- an[term, "Pr(>Chisq)"]
      r_es <- sqrt(chi2 / (chi2 + nrow(sub)))
      logger$log(sprintf("  %-35s Chi2(%d) = %6.3f,  p = %.4f,  r = %.3f",
                         term, df_t, chi2, p, r_es))
    }
  } else {
    # F-test with Satterthwaite df from lmerTest
    an <- anova(fit, type = "III")
    terms_of_interest <- c("starting_vertex")
    for (term in terms_of_interest) {
      if (!term %in% rownames(an)) next
      f_val  <- an[term, "F value"]
      df1    <- an[term, "NumDF"]
      df2    <- an[term, "DenDF"]
      p      <- an[term, "Pr(>F)"]
      eta2_p <- (f_val * df1) / (f_val * df1 + df2)
      logger$log(sprintf("  %-35s F(%d, %6.1f) = %6.3f,  p = %.4f,  η²p = %.3f",
                         term, df1, df2, f_val, p, eta2_p))
    }
    # ICC from random effects
    re       <- as.data.frame(VarCorr(fit))
    var_part <- re[re$grp == "participant_ID", "vcov"]
    var_res  <- re[re$grp == "Residual",       "vcov"]
    logger$log(sprintf("  ICC (participant) = %.3f", var_part / (var_part + var_res)))
  }
  logger$log("")
}

# ── Run all three outcomes for one experiment ─────────────────────────────────

run_geometry_lmm <- function(data, seq_names, out_dir, exp_label) {
  logger   <- make_logger()
  date_str <- format(Sys.time(), "%d-%m-%Y %H:%M")

  logger$log(sprintf("LMM/GLMM: Starting Vertex Effect (omnibus) — %s", exp_label))
  logger$log(sprintf("Generated:    %s", date_str))
  logger$log(sprintf("Input script: %s", script_file))
  logger$log(sprintf("Output dir:   %s", out_dir))
  logger$log(sprintf("Model:        outcome ~ starting_vertex + (1|participant_ID)"))
  logger$log(sprintf("Note:         Continuous outcomes (DL, RT) use lmer + Satterthwaite F-test."))
  logger$log(sprintf("              Binary outcome (error rate) uses glmer + Wald chi-square."))
  logger$log(strrep("=", 60))
  logger$log("")

  outcomes <- list(
    list(col = "distance_dl",   label = "DL distance        (lmer, continuous)"),
    list(col = "fail",          label = "Error rate / fail  (glmer, binomial)"),
    list(col = "response_time", label = "Response time / ms (lmer, continuous)")
  )

  for (o in outcomes) {
    logger$log(strrep("-", 60))
    logger$log(sprintf("Outcome: %s", o$label))
    logger$log(strrep("-", 60))
    run_omnibus(data, seq_names, o$col, logger)
  }

  report_path <- file.path(out_dir,
    sprintf("%s_geometry_lmm_report.txt", format(Sys.Date(), "%d-%m-%Y")))
  logger$save(report_path)
}

# ── Run ───────────────────────────────────────────────────────────────────────

run_geometry_lmm(df, seq_names_exp1, out_dir_exp1, "Experiment 1")
run_geometry_lmm(df, seq_names_exp2, out_dir_exp2, "Experiment 2")
