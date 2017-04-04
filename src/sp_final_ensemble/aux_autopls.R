# train a model and/or assess its performance
train_cv_pls <- function(
  training, validation = NULL, k, K, eta, 
  metrics = list(logLoss = Metrics::logLoss, 
                 auc = Metrics::auc)) {
  
  # grid search
  message("Performing grid search with ", k, " folds...")
  mod_splscv <- spls::cv.splsda(
    x = as.matrix(training[, !"cancer"]), 
    y = training$cancer, 
    k = k, 
    K = K, 
    eta = eta, 
    classifier = "logistic", 
    n.core = 4
  )
  # real model
  mod_spls <- spls::splsda(
    x = as.matrix(training[, !"cancer"]), 
    y = training$cancer, 
    K = mod_splscv$K.opt, 
    eta = mod_splscv$eta.opt, 
    classifier = "logistic")
  
  if (is.null(validation)) return(mod_spls)
  
  ypred <- spls::predict.splsda(
    mod_spls, 
    validation[, !"cancer"], 
    fit.type = "response"
  )
  yval <- validation$cancer
  
  # quality metrics
  ans <- plyr::llply(
    metrics, 
    function(met) {
      do.call(met, list(yval, ypred))
    }
  )
  
  ll <- Metrics::ll(validation$cancer, ypred)
  # browser()
  data.frame(ans, patientid = rownames(validation), 
             cancer = validation$cancer, predicted = ypred, 
             ll = ll, stringsAsFactors = FALSE)
}

# wrapper for estimating performance and adjusting final model
wrapper_pls <- function(
  df_model, 
  cv_out_k = 5, 
  cv_out_times = 1,
  cv_in_k = 5, 
  K = 1:8, 
  eta = seq(from = .6, to = .95, by = .05)
)  {
  ans <- list()
  if (!is.null(cv_out_times)) {
    list_cv <- caret::createMultiFolds(df_model$cancer, k = cv_out_k, times = cv_out_times)
    
    message("Estimating model performance...")
    
    df_ans <- plyr::ldply(
      list_cv, 
      function(lst) {
        training <- df_model[ lst, !("patientid")]
        validation <- df_model[-lst, !("patientid")]
        # browser()
        rownames(validation) <- df_model[-lst, patientid]
          
        val_pls <- train_cv_pls(training = training, validation = validation, K = K, eta = eta, k = cv_in_k)
      }, .id = "fold", .progress = "text"
    )
    ans$perf <- df_ans
  }
  
  mod_pls <- train_cv_pls(training = df_model[, !("patientid")], K = K, eta = eta, k = cv_in_k)
  ans$model <- mod_pls
  
  return(ans)
}

# wrapper whole submission
# 
wrapper_submit_pls <- function(
  df_features, 
  df_cancer, 
  df_submit, 
  out_submit = "model_1",
  out_perf = paste0(out_submit, "_perf"), 
  ...
) {
  # join features and cancer - training examples only
  df_model <- df_features[df_cancer, on = "patientid", nomatch = 0]
  
  # model
  df_out <- wrapper_pls(df_model, ...)
  write.csv(
    df_out$perf, file = paste0(out_perf, ".csv"), 
    quote = FALSE, row.names = FALSE)
  
  # submission
  df_eval <- df_features[(patientid %in% df_submit$patientid)]
  pred_out <- spls::predict.splsda(
    df_out$model, 
    newx = df_eval[, !"patientid"], 
    fit.type = "response"
  )
  names(pred_out) <- df_eval$patientid
  
  # na patients
  NA_prop <- mean(df_cancer$cancer)
  pat_na <- setdiff(df_submit$patientid, df_eval$patientid)
  v_na <- setNames(rep(NA_prop, length(pat_na)), pat_na)
  
  v_out <- c(pred_out, v_na)
  df_write <- data.frame(
    id = names(v_out), 
    cancer = v_out, 
    stringsAsFactors = FALSE
  )
  
  write.csv(
    df_write, file = paste0(out_submit, ".csv"), 
    quote = FALSE, row.names = FALSE)
  
  return(df_out)
}
