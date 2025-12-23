set.seed(123)

########### 1) LOAD DATA ###########
train_path <- "2020-2024 case data-CSV1TRAIN.csv"
test_path  <- "2025 case data-CSV2.csv"

csv1 <- readr::read_csv(train_path, show_col_types = FALSE)


csv2 <- readr::read_csv(test_path,  show_col_types = FALSE)
csv2 <- csv2 %>% mutate(row_id = row_number())


########### 2) DEFINE COLUMNS ###########
# Target in training data
target <- "decisionDirection"  # 1=Conservative, 2=Liberal

# Predictors (edit this list only if your column names differ)
predictors <- c(
  "petitioner",
  "respondent",
  "issueArea",
  "issue",
  "lawType",
  "lcDispositionDirection",
  "jurisdiction"
)

# Keeping needed columns
train_df <- csv1 %>%
  dplyr::select(dplyr::all_of(c(target, predictors)))

test_df <- csv2 %>%
  dplyr::select(row_id, dplyr::all_of(predictors))


########### 3) CLEAN AND ALIGN FACTOR LEVELS ###########
# Ensures target is a factor with levels 1 and 2
train_df[[target]] <- factor(train_df[[target]], levels = c(1, 2))

# Convert predictor columns to factors (categorical codes)
for (col in predictors) {
  train_df[[col]] <- factor(train_df[[col]])
  test_df[[col]]  <- factor(test_df[[col]])
}

# Align factor levels in test
for (col in predictors) {
  test_df[[col]] <- factor(test_df[[col]], levels = levels(train_df[[col]]))
}
bad_rows <- test_df %>%
  mutate(bad = if_any(all_of(predictors), is.na)) %>%
  filter(bad)

bad_rows


train_df <- train_df %>% tidyr::drop_na()


########### 4) TRAIN DECISION TREE W CV ###########
ctrl <- caret::trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = caret::twoClassSummary,
  savePredictions = "final"
)

train_df <- train_df %>%
  mutate(decisionDirection = ifelse(decisionDirection == 1, "Conservative", "Liberal")) %>%
  mutate(decisionDirection = factor(decisionDirection, levels = c("Conservative", "Liberal")))

# Train
tree_fit <- caret::train(
  decisionDirection ~ .,
  data = train_df,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 20
)


########### 5) REGULARIZED LOGISTIC REGRESSION ###########
glmnet_fit <- caret::train(
  decisionDirection ~ .,
  data = train_df,
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 20
)
print(glmnet_fit)


########### 6) Model comparison ###########
resamps <- resamples(list(
  DecisionTree = tree_fit,
  Logistic = glmnet_fit
))

summary(resamps)


bwplot(resamps, metric = "ROC")


print(tree_fit)


rpart.plot::rpart.plot(tree_fit$finalModel, main = "Decision Tree (rpart)")


########### 7) CONFUSION MATRIX ON TRAINING DATA ###########
train_preds <- predict(tree_fit, newdata = train_df)

confusionMatrix(
  train_preds,
  train_df$decisionDirection,
  positive = "Conservative"
)


roc_obj <- roc(
  response = train_df$decisionDirection,
  predictor = predict(tree_fit, train_df, type = "prob")[, "Conservative"]
)


plot(roc_obj,
     main = "ROC Curve – Decision Tree Model",
     col = "blue",
     lwd = 2)


varImpPlot <- varImp(tree_fit)
plot(varImpPlot, main = "Variable Importance (rpart)")


########### 8) PREDICT FOR CSV2 ###########
# Predict class and probabilities
good_test <- test_df %>%
  filter(!if_any(all_of(predictors), is.na))

pred_class <- predict(tree_fit, newdata = good_test)
pred_prob  <- predict(tree_fit, newdata = good_test, type = "prob")

pred_code <- ifelse(pred_class == "Conservative", 1, 2)

pred_out <- good_test %>%
  select(row_id) %>%
  mutate(
    predictedDecisionDirection = pred_code,
    probConservative = pred_prob$Conservative,
    probLiberal = pred_prob$Liberal
  )


########### 9) Output: attach predictions to csv2 and write a file ###########
out <- csv2 %>%
  left_join(pred_out, by = "row_id")

out %>%
  filter(!is.na(predictedDecisionDirection)) %>%
  ggplot(aes(x = probConservative)) +
  geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7) +
  labs(
    title = "Predicted Probability of Conservative Rulings (2025 Cases)",
    x = "P(Conservative)",
    y = "Number of Cases"
  )


out %>% ## Lower Court vs Supreme Court Direction
  filter(!is.na(predictedDecisionDirection)) %>%
  ggplot(aes(
    x = factor(lcDispositionDirection),
    fill = factor(predictedDecisionDirection)
  )) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Lower Court Ideology vs Predicted Supreme Court Outcome",
    x = "Lower Court Direction (1=Conservative, 2=Liberal)",
    fill = "Predicted SCOTUS Direction"
  )


# Save results
write.csv(out, "CSV2_with_predictions.csv", row.names = FALSE)

# print preview
print(out %>% dplyr::select(dplyr::any_of(c("docket", "caseName")),
        predictedDecisionDirection, probConservative, probLiberal))
