#LIBRARIES
library(data.table)
library(mlr)
library(parallelMap)
library(ggplot2)
# Constants
extreme_values <<- c(0,511)

generateModel <- function(model_family) {
  
  lrn = makeLearner(
    model_family,
    predict.type = "prob"
  )
  ps = makeParamSet()
  if(model_family == "classif.xgboost") {
    lrn = makeLearner(
      "classif.xgboost",
      predict.type = "prob",
      eval_metric = "logloss",
      nthread = 7,
      eta = 0.01,
      nrounds = 1000,
      colsample_bytree = 0.9,
      subsample = 0.85,
      min_child_weight = 96
    )
    ps = makeParamSet(
      makeNumericParam("eta", lower = 0.001, upper = 0.3),
      makeIntegerParam("nrounds",lower = 1, upper = 100)
      #makeNumericParam("colsample_bytree", lower = 1, upper = 2, trafo = function(x) x/2),
      #makeNumericParam("subsample", lower = 1, upper = 2, trafo = function(x) x/2)
    )
  }
  if (model_family == "classif.gbm") {
    lrn = makeLearner(
      "classif.gbm",
      predict.type = "prob",
            # n.trees = 512,
            # interaction.depth = 5,
            # shrinkage = 0.0005728955
          #  minobsinnode = 1
      n.trees = 10000,
      # n.minobsinnode = 10, # eliminar
      # interaction.depth = 5, #eliminar
      #interaction.depth = 3,
      shrinkage = 0.001
    )
    
    ps = makeParamSet(
      #makeIntegerParam("n.trees", lower = 1, upper = 700),
      makeIntegerParam("interaction.depth", lower = 1, upper = 10),
      #makeNumericParam("shrinkage",lower = 0, upper = 0.001),
      makeIntegerParam("n.minobsinnode", lower = 1, upper = 10)
    )
  }
  if(model_family == "classif.ranger") {
    lrn = makeLearner(
      "classif.ranger",
      predict.type = "prob",
      num.trees = 50
    )
    ps = makeParamSet(
      makeIntegerParam("num.trees", lower = 1, upper = 500),
      makeIntegerParam("mtry", lower = 1, upper = 15),
      makeNumericParam("sample.fraction",lower = 1, upper = 2, trafo = function(x) x/2)
    )
  }
  if(model_family == "classif.dbnDNN"){
    lrn = makeLearner(
      model_family,
      predict.type = "prob",
      hidden = c(50,5,2,1)
    )

  }
  if(model_family == "classif.plsdaCaret"){
    lrn = makeLearner(
      model_family,
      predict.type = "prob",
      ncomp = 6,
      probMethod = "softmax"
      #eta = 0.7
    )
  }
  
  return(list(lrn = lrn,ps = ps))
}
predict_object <- function(tr,test_task){
  if(sum(class(test_task) %in% c("data.table","data.frame")) >= 1){
    preds <- predict(tr, newdata=data.frame(test_task))
  } else {
    preds <- predict(tr,test_task)
  }
  preds <- data.table(preds$data)[,prob.1]
  return(preds)
}
predictCv <- function(tr, test_task) {
  preds <- 0
  if(is.null(tr$models)) {
    preds <- predict_object(tr,test_task)
  } else {
    n_cv <- length(tr$models)
    for(i in 1:n_cv) { 
      pred_i <- predict_object(tr$models[[i]],test_task)
      preds <- preds + pred_i
    }
    preds <- preds/n_cv
  }
  # if(sum(class(test_task) %in% c("data.frame","data.table")) >= 1){
  #   
  # for(i in 1:n_cv) { 
  #   pred = predict(tr$models[[i]], newdata=data.frame(score))
  #   pred_i <- data.table(pred$data)[,prob.1]
  #   preds <- preds + pred_i
  # }
  # preds <- preds/n_cv# 
  
  return(preds)
}

LogLossBinary = function(actual, predicted) {
  eps = 1e-15
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

my.AUC <- function (actual, predicted) {
  decimals = 6
  predicted <- round(predicted, decimals)
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos *  n_neg)
  return(auc)
}



