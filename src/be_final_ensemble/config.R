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
      eta = 0.03757,
      nrounds = 2500,
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
      n.trees = 474,
      interaction.depth = 3,
      shrinkage = 0.004618582
    )
    
    ps = makeParamSet(
      makeIntegerParam("n.trees", lower = 1, upper = 700),
      makeIntegerParam("interaction.depth", lower = 1, upper = 10),
      makeNumericParam("shrinkage",lower = 0, upper = 0.001),
      makeIntegerParam("n.minobsinnode", lower = 1, upper = 20)
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



## EXTRA FUNCTIONS----------------------------------------------------------------------------------
aggregate_patient <- function(dt) {
  dt <- na_to_zeros(dt,c(
    "diameter",
    "max_intensity",
    "min_intensity",
    "mean_intensity",
    "nodule_pred",
    "diameter_patches",
    "score_patches")
  )
  
  final_df = dt[,.(total_nodules_unet=sum(!is.na(x)), 
                   total_nodules_patches = sum(!is.na(x_patches)),
                   big_nodules_unet = sum(diameter > 20,na.rm=T),
                   big_nodules_patches = sum(diameter_patches > 20,na.rm = T),
                   max_diameter = max(diameter,na.rm = T),
                   max_diameter_patches = max(diameter_patches,na.rm = T),
                   num_slices_unet = uniqueN(nslice[!is.na(x)]),
                   num_slices_patches = uniqueN(nslice[!is.na(x_patches)]),
                   nodules_per_slice_unet = sum(!is.na(x))/uniqueN(nslice[!is.na(x)]),
                   nodules_per_slice_patches = sum(!is.na(x_patches))/uniqueN(nslice[!is.na(x_patches)]),
                   max_intensity = max(max_intensity, na.rm=T), 
                   max_mean_intensity = max(mean_intensity, na.rm=T),
                   min_intensity = min(min_intensity,na.rm=T),
                   max_score = max(nodule_pred, na.rm=T),
                   mean_score = mean(nodule_pred),
                   max_score_patches = max(score_patches,na.rm=T),
                   mean_score_patches = mean(score_patches,na.rm=T)
                   
  ),
  by=.(patientid)]
  
  final_df[!is.finite(max_intensity), max_intensity:=0]
  final_df[!is.finite(min_intensity), min_intensity:=0]
  final_df[!is.finite(max_mean_intensity), max_mean_intensity:=0]
  final_df[is.na(final_df)] <- 0
  final_df[max_score < 0,max_score := 0]
  
  # Computing if the patient has consecutive nodules 
  dt_2d <- dt[
    nodule_pred > 0.2 & max_intensity > 0.96,
    .(patientid,nslice,x,y)]
  dt_2d_bis <- dt[
    nodule_pred > 0.2 & max_intensity > 0.96,
    .(patientid,nslice2 = nslice,x2 = x,y2 = y)]
  dt_3d <- merge(dt_2d,dt_2d_bis,all.x = T, by = "patientid",allow.cartesian = TRUE)
  dt_3d <- dt_3d[nslice2 > nslice]
  setkey(dt_3d,patientid,nslice,nslice2)
  dt_3d <- dt_3d[,.SD[1],c("patientid","nslice")]
  dt_3d <- dt_3d[nslice2-nslice < 3]
  dt_3d[,d_nodule := abs(x-x2)+abs(y-y2) < 10]
  nodules_consec <- dt_3d[,.(consec_nods = sum(d_nodule)),patientid]
  
  final_df <- merge(final_df,nodules_consec,all.x = T, by = "patientid")
  final_df[is.na(consec_nods),consec_nods := 0]
  
  
  # Computing if the patient has consecutive nodules of patches
  dt_2d <- dt[!is.na(x_patches),.(patientid,nslice,x = x_patches,y = y_patches)]
  dt_2d_bis <- dt[!is.na(x_patches),.(patientid,nslice2 = nslice,x2=x_patches,y2=y_patches)]
  dt_3d <- merge(dt_2d,dt_2d_bis,all.x=T,by="patientid",allow.cartesian = TRUE)
  dt_3d <- dt_3d[nslice2 > nslice]
  setkey(dt_3d,patientid,nslice,nslice2)
  dt_3d <- dt_3d[,.SD[1],c("patientid","nslice")]
  dt_3d <- dt_3d[nslice2-nslice <3]
  dt_3d[,d_nodule := abs(x-x2) + abs(y-y2) < 10]
  nodules_patches_consec <- dt_3d[,.(consec_nods_patches = sum(d_nodule)),patientid]
  
  final_df <- merge(final_df,nodules_patches_consec,all.x=T,by="patientid")
  final_df[is.na(consec_nods_patches),consec_nods_patches:=0]
  
  # Computing the variables of the nodules with highter score for patient
  dt[,`:=`(
    max_score = max(nodule_pred,na.rm=T),
    max_score_patches=max(score_patches,na.rm=T)),patientid]
  max_score <- dt[
    (max_score == nodule_pred & max_score > 0),
    .(patientid,
      diameter_nodule = diameter,
      max_intensity_nodule = max_intensity,
      nslice_nodule = nslice,
      mean_intensity_nodule = mean_intensity)
    ]
  
  max_score_nodule <- dt[
    max_score_patches == score_patches & max_score_patches > 0,
    .(patientid,
      nslice_nodule_patch = nslice,
      diameter_nodule_patch = diameter_patches)
    ]
  max_score_nodule <- max_score_nodule[,.SD[1],patientid]
  final_df <- merge(final_df,max_score,all.x = T, by="patientid")
  final_df <- merge(final_df,max_score_nodule,all.x=T,by = "patientid")
  final_df <- na_to_zeros(
    final_df,
    c("diameter_nodule",
      "max_intensity_nodule",
      "nslice_nodule",
      "mean_intensity_nodule",
      "nslice_nodule_patch",
      "diameter_nodule_patch")
  )
  return(final_df)
}
na_to_zeros <- function(dt,name_vars) {
  for(name_var in name_vars) {
    setnames(dt,name_var,"id")
    if(nrow(dt[is.na(id)]) > 0) {
      dt[is.na(id),id := 0]
    }
    setnames(dt,"id",name_var)
  }
  return(dt)
}