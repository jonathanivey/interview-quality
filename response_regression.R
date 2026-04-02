# 1. Load required libraries
library(lme4)
library(lmerTest)
library(readr)
library(car)
library(sjPlot)


# 2. Load analysis data
df <- read_csv("data/analysis/final_data.csv")


# 3. Fit the Mixed Effects Model
# We use (1 | project_id / source) to indicate that sources are nested within projects
model <- lmer(
  quality_criterion ~ 
    rq_relevance + 
    attributed_meaning + 
    specificity + 
    n_tokens + 
    immediate_relevance + 
    spontaneity + 
    self_reportedness + 
    average_surprisal + 
    token_ratio + 
    clarity + 
    (1 | project_id / source), 
  data = df
)


vif(model)

# Generates a clean HTML table in your Viewer pane
tab_model(model, show.std = TRUE, digits = 3)
