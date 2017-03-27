library(data.table)
library(pls)
library(spls)
# library(mixOmics)
library(plyr)
library(dplyr)
library(caret)
# library(mlr)
library(Metrics)

set.seed(1)

# submission df
# patient format is bad - will be improved
df_submit <- read.csv("../../data/stage1_sample_submission.csv", stringsAsFactors = FALSE)
colnames(df_submit) <- c("patientid", "cancer")

# labels
df_cancer <- read.csv("../../data/stage1_labels.csv", stringsAsFactors = FALSE)
colnames(df_cancer) <- c("patientid", "cancer")

# df_nodule <- fread("../../data/output_dl_example_augmented.csv")
df_nodule <- fread("submissions/sp_01_features.csv")

# inner join, add cancer column
df_model <- df_nodule[df_cancer, on = "patientid", nomatch = 0]
#df_model$cancer <- as.factor(ifelse(df_model$cancer, "yes", "no"))


# Split train-test
set.seed(2)
inTraining <- caret::createDataPartition(df_model$cancer, p = .85, list = FALSE)
# lapply(row_train, length)
training <- df_model[ inTraining, !("patientid"), with = FALSE]
testing  <- df_model[-inTraining, !("patientid"), with = FALSE]


cv_X_train <- as.matrix(training[, !"cancer"])
cv_y_train <- training$cancer
cv_X_test <- as.matrix(testing[, !"cancer"])
cv_y_test <- testing$cancer
cv_X <- as.matrix(df_model[, !c("patientid", "cancer")])
cv_y <- df_model$cancer

library(xgboost)

xgb_grid_1 = expand.grid(
  nrounds = 50,
  eta = c(1, 0.1, 0.01),
  max_depth = c(2, 3),
  gamma = c(1, 2, 3, 5),
  colsample_bytree = c(.7, .5),
  min_child_weight = c(15, 30, 50),
  subsample = c(.7, .5)
)

########### Estimate logloss ########### 

# 
# # pack the training control parameters
# source("aux_LogLossSummary.R")
# set.seed(4)
# xgb_trcontrol_1 = trainControl(
#   method = "cv",
#   number = 5,
#   verboseIter = TRUE,
#   returnData = FALSE,
#   returnResamp = "all",                                                        # save losses across all models
#   classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
#   # summaryFunction = LogLossSummary,
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # train the model for each parameter combination in the grid, 
# #   using CV to evaluate
# xgb_train_1 = train(
#   x = cv_X_train, 
#   y = as.factor(ifelse(cv_y_train, "Cancer", "NoCancer")), 
#   trControl = xgb_trcontrol_1,
#   tuneGrid = xgb_grid_1,
#   method = "xgbTree", 
#   metric = "logLoss"
# )
# 
# best_param <- as.list(xgb_train_1$bestTune)
# other_param <- list(
#   early_stopping_rounds = 15 
# )
# 
# bst <- do.call(
#   xgboost, 
#   c(list(data = cv_X_train, label = (cv_y_train), objective = "binary:logistic"), 
#     best_param, other_param)
# )
# 
# summary(bst)
# Metrics::auc(
#   cv_y_test, 
#   predict(bst, cv_X_test)
# )
# Metrics::logLoss(
#   cv_y_test, 
#   predict(bst, cv_X_test)
# )
# importance_matrix <- xgb.importance(colnames(cv_X_train), model = bst)
# xgboost::xgb.plot.importance(head(importance_matrix, 20))

########### Train with all the data ########### 
# pack the training control parameters
set.seed(5)
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  # summaryFunction = LogLossSummary,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train_1 = train(
  x = cv_X, 
  y = as.factor(ifelse(cv_y, "Cancer", "NoCancer")), 
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree", 
  metric = "logLoss"
)

best_param <- as.list(xgb_train_1$bestTune)
other_param <- list(
  early_stopping_rounds = 20 
)

set.seed(6)
bst <- do.call(
  xgboost, 
  c(list(data = cv_X_train, label = (cv_y_train), objective = "binary:logistic"), 
    best_param, other_param)
)

# save results
df_eval <- df_nodule[patientid %in% df_submit$patientid]
pred_out <- predict(
  bst, 
  newdata = as.matrix(df_eval[, !"patientid"])
)
names(pred_out) <- df_eval$patientid

# background proportion for missing patient imputation
NA_prop <- mean(df_cancer$cancer)
pat_na <- setdiff(df_submit$patientid, df_eval$patientid)
v_na <- setNames(rep(NA_prop, length(pat_na)), pat_na)

v_out <- c(pred_out, v_na)
df_out <- data.frame(
  id = names(v_out), 
  cancer = v_out, 
  stringsAsFactors = FALSE
)
write.csv(df_out, file = "submissions/sp_03_xgboost.csv", row.names = FALSE, quote = FALSE)
