############################################################
# Decision Tree model for Supreme Court outcomes (SCDB)
# Train: 1946-001-01 to 2019-074-01
# Test : 2020-001-01 to 2024-068-01
############################################################
############################################
######### 1) PACKAGES #########
library(tidyverse)


library(rpart)
library(rpart.plot)
library(caret)


library(pROC)


library(janitor)


library(scales)


library(readr)
library(ranger)

CaseCenteredData <- read_csv("Desktop/Data final/CaseCenteredData.csv")


set.seed(123)

######### 2) KEY COLUMNS #########
docket_col <- "docketId"
y_col <- "decisionDirection"   # Conservative vs Liberal

if (!all(c(docket_col, y_col) %in% names(CaseCenteredData))) {
  stop("Required columns not found. Check column names with names(CaseCenteredData).")
}

######### 3) TRAIN/TEST SPLIT BY docketId #########
train_start <- "1946-001-01"
train_end   <- "2019-074-01"
test_start  <- "2020-001-01"
test_end    <- "2024-068-01"

CaseCenteredData2 <- CaseCenteredData %>%
  dplyr::filter(decisionDirection %in% c(1, 2)) %>%
  dplyr::mutate(
    y = factor(ifelse(decisionDirection == 1, "Conservative", "Liberal"))
  )


train <- CaseCenteredData2 %>%
  filter(docketId >= train_start, docketId <= train_end)

test <- CaseCenteredData2 %>%
  filter(docketId >= test_start, docketId <= test_end)

cat("Training rows:", nrow(train), "\n")


cat("Test rows:", nrow(test), "\n")


######### 4) OUTCOMES/PREDICTORS #########
train <- train %>% filter(!is.na(decisionDirection))
test  <- test  %>% filter(!is.na(decisionDirection))

# Remove leakage / IDs
drop_cols <- c(
  "docketId", "decisionDirection",
  "voteId", "caseId", "caseIssuesId",
  "majVotes", "minVotes", "majOpinWriter",
  "lcDisposition", "lcDispositionDirection",
  "dateDecision"   # <-- ADD THIS
)
predictors <- setdiff(names(train), drop_cols)
predictors <- intersect(predictors, names(test))

# DROP test rows that became NA (class not seen in training)
before <- nrow(test)
test <- test %>% filter(!is.na(y))
cat("Dropped", before - nrow(test), "test rows with unseen outcome class.\n")


######### 5) PREPARE MODELING FRAMES #########
prep <- function(dat) {

  out <- dat %>%
    dplyr::select(y, all_of(predictors)) %>%
    dplyr::mutate(dplyr::across(where(is.character), as.factor)) %>%
    dplyr::mutate(dplyr::across(where(is.numeric), function(x) {
      x[is.na(x)] <- median(x, na.rm = TRUE)
      x
    }))

  # Convert NA levels for predictor factors WITHOUT across()
  fac_cols <- setdiff(names(out)[sapply(out, is.factor)], "y")
  out[fac_cols] <- lapply(out[fac_cols], function(x) {
    forcats::fct_na_value_to_level(x, level = "MISSING")
  })
  out
}

# Test checks
cat("Train outcome levels:\n"); print(levels(train$y))


cat("Test outcome counts:\n"); print(table(test$y, useNA = "ifany"))


train_m <- prep(train)
test_m  <- prep(test)

# Align train/test predictors
train_x <- train_m %>% dplyr::select(-y)
train_y <- train_m$y

test_x  <- test_m %>% dplyr::select(-y)
test_y  <- test_m$y   # keep for evaluation

train_x <- as.data.frame(train_x)
test_x  <- as.data.frame(test_x)

# Add missing columns to test
missing_cols <- setdiff(names(train_x), names(test_x))
for (m in missing_cols) test_x[[m]] <- NA

# Drop extra columns from test
extra_cols <- setdiff(names(test_x), names(train_x))
if (length(extra_cols) > 0) test_x <- test_x %>% dplyr::select(-all_of(extra_cols))

# Put columns in identical order
test_x <- test_x[, names(train_x)]

# Match factor levels between train and test
for (nm in names(train_x)) {
  if (is.factor(train_x[[nm]])) {
    test_x[[nm]] <- factor(test_x[[nm]], levels = levels(train_x[[nm]]))
  }
}

# Remove near-zero variance predictors
nzv <- caret::nearZeroVar(train_x)
if (length(nzv) > 0) {
  train_x <- train_x[, -nzv, drop = FALSE]
  test_x  <- test_x[,  -nzv, drop = FALSE]
}

# Drop very high-cardinality factors
drop_high_cardinality <- function(df, max_levels = 50) {
  facs <- names(df)[sapply(df, is.factor)]
  too_many <- facs[sapply(df[facs], nlevels) > max_levels]
  df[, setdiff(names(df), too_many), drop = FALSE]
}

train_x <- drop_high_cardinality(train_x, max_levels = 50)
test_x  <- test_x[, names(train_x), drop = FALSE]  # keep exact same cols


######### 6) FIT DECISION TREE #########
# Fit
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

cp_grid <- expand.grid(
  cp = seq(0.0005, 0.02, length.out = 20)
)

# Make sure x/y are plain data.frames
train_x <- as.data.frame(train_x)
test_x  <- as.data.frame(test_x)

# Ensures factor outcome is plain factor
train_y <- as.factor(train_y)
test_y  <- as.factor(test_y)

tree_fit <- suppressWarnings(
  suppressMessages(
    caret::train(
      x = train_x, y = train_y,
      method = "rpart",
      metric = "ROC",
      trControl = ctrl,
      tuneGrid = cp_grid,
      control = rpart.control(xval = 10)   # <-- add this
    )
  )
)

tree <- tree_fit$finalModel


pred_class <- predict(tree_fit, newdata = test_x, type = "raw")
pred_prob  <- predict(tree_fit, newdata = test_x, type = "prob")
cm <- caret::confusionMatrix(pred_class, test_y)
cm


# Make sure test has the same predictor columns as train
train_x <- train_m %>% dplyr::select(-y)
test_x  <- test_m  %>% dplyr::select(-y)

# Add any missing columns to test as NA
missing_cols <- setdiff(names(train_x), names(test_x))
for (m in missing_cols) test_x[[m]] <- NA

# Drop any extra columns in test
extra_cols <- setdiff(names(test_x), names(train_x))
test_x <- test_x %>% dplyr::select(-all_of(extra_cols))

# Put columns in the same order
test_x <- test_x[, names(train_x)]

# Predict
pred_class <- predict(tree_fit, newdata = test_x)
pred_prob  <- predict(tree_fit, newdata = test_x, type = "prob")

# Only request probs if the model is truly classification with >=2 classes
if (length(tree$ylevels) >= 2) {
  pred_prob <- predict(tree, newdata = test_m, type = "prob")
}

tree <- tree_fit$finalModel

cm <- confusionMatrix(pred_class, test_m$y)
print(cm)


######### 7) PLOTS #########
# Caret's tuning/CV plot for cp
plot(tree_fit)


tree_xval <- rpart::rpart(
  y ~ .,
  data = train_m,
  method = "class",
  control = rpart::rpart.control(xval = 10, cp = 0)
)

plotcp(tree_xval)


printcp(tree_xval)


# Tree visualization
suppressWarnings(
  rpart.plot::rpart.plot(tree, type = 2, extra = 104, fallen.leaves = TRUE,
                         main = "Pruned Decision Tree: Supreme Court Decisions")
)


roundint=FALSE

# Tree visualization
suppressWarnings(
  rpart.plot(
    tree,
    type = 2,
    extra = 104,
    fallen.leaves = TRUE,
    main = "Pruned Decision Tree: Supreme Court Decisions"
  )
)

# Variable importance
imp <- tibble(
  variable = names(tree$variable.importance),
  importance = as.numeric(tree$variable.importance)
) %>% arrange(desc(importance))

ggplot(imp %>% slice_head(n = 15),
       aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top Predictors of Supreme Court Decisions",
       x = NULL, y = "Importance")


# Confusion matrix heatmap
as.data.frame(cm$table) %>%
  ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  labs(title = "Confusion Matrix: 2020–2024 Test Set")


# ROC curve
if (nlevels(test_m$y) == 2) {
  roc_obj <- roc(test_m$y, pred_prob[,2])
  plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc(roc_obj), 3), ")"))
}


coords_best <- coords(
  roc_obj,
  x = "best",
  best.method = "youden",
  transpose = FALSE
)
coords_best


best_thresh <- coords_best$threshold

pred_class_opt <- factor(
  ifelse(pred_prob[, "Liberal"] > best_thresh, "Liberal", "Conservative"),
  levels = levels(test$y)
)

confusionMatrix(pred_class_opt, test$y)


rf_fit <- ranger(
  y ~ .,
  data = train_m,
  probability = TRUE,
  num.trees = 500,
  mtry = floor(sqrt(ncol(train_m))),
  min.node.size = 10,
  importance = "impurity"
)

rf_prob <- predict(rf_fit, test_m)$predictions[, "Liberal"]

roc_rf <- roc(test$y, rf_prob)


auc(roc_rf)


######### 8) CONSERVATIVE SHIFT ANALYSIS #########
results <- tibble::tibble(
  actual = test_y,
  predicted = pred_class
)

actual_cons <- mean(results$actual == "Conservative", na.rm = TRUE)
pred_cons   <- mean(results$predicted == "Conservative", na.rm = TRUE)

share_df <- tibble::tibble(
  Type = c("Actual", "Predicted"),
  Share = c(actual_cons, pred_cons)
)

ggplot2::ggplot(share_df, ggplot2::aes(Type, Share)) +
  ggplot2::geom_col() +
  ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                              limits = c(0, 1)) +
  ggplot2::geom_text(ggplot2::aes(label = scales::percent(Share, accuracy = 0.1)),
                     vjust = -0.4) +
  ggplot2::labs(title = "Conservative Share in Supreme Court Decisions (2020–2024)",
                y = "Share", x = NULL)


cat("Actual Conservative Share:", round(actual_cons, 3), "\n")


cat("Predicted Conservative Share:", round(pred_cons, 3), "\n")


#Packages
library(readr)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)
library(glmnet)
