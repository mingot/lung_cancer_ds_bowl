## REPO PATH, CANVIAR SI ES NECESSARI
path_repo <<- "D:/lung_cancer_ds_bowl/"
path_data <<- "D:/output/"
# IMPORTS ------------------------------------------------------------------------------------------
source(paste0(path_repo,"src/be_final_ensemble/config.R"))
source(paste0(path_repo,"src/be_final_ensemble/fp_model.R"))
source(paste0(path_repo,"src/be_final_ensemble/aggregate_dt.R"))

# DATA ---------------------------------------------------------------------------------------------

## PATIENTS AND LABELS Data ------------------------------------------------------------------------

## Add variables to all sets

dataset_final <- generate_patient_dt(path_repo)
# SEPARATING TRAIN AND SCORING ---------------------------------------------------------------------
patients_train <- dataset_final[dataset == "training",patientid]
dataset_final[,dataset := NULL]
dataset_final[,consec_nods := NULL]
dataset_final[,score_2_patch := NULL]
dataset_final[,max_score_2 := NULL]
features_sp <- fread(paste0(path_repo,"src/sp_final_ensemble/submissions/sp_01_features.csv"))
dataset_final_m <- merge(dataset_final,features_sp,all.x = T,by = "patientid")
dataset_final_m <- na_to_zeros(dataset_final_m,names(dataset_final_m))
nombres_m <- names(dataset_final_m)
nombres_m <- nombres_m[nombres_m!= "patientid"]
dataset_final_m[,cancer := as.numeric(as.character(cancer))]
dataset_final_m <- na_to_zeros(dataset_final_m,names(dataset_final_m))
for(n in nombres_m) {
  if(var(dataset_final_m[[n]]) < 0.0001) dataset_final_m[[n]] <- NULL
}


df_model <- dataset_final_m
# Split train-test
set.seed(2)
inTraining <- caret::createDataPartition(df_model$cancer, p = .85, list = FALSE)
# lapply(row_train, length)
training <- df_model[ inTraining, !("patientid"), with = FALSE]
testing  <- df_model[-inTraining, !("patientid"), with = FALSE]

NA_prop <- training[,mean(as.numeric(as.character(cancer)))]
# To assess performance, cv in train
set.seed(3)
mod_splscv <- spls::cv.splsda(
  x = as.matrix(training[, !"cancer"]), 
  y = training$cancer, 
  K = 1:8, 
  eta = seq(from = .6, to = .95, by = .05), 
  classifier = "logistic", 
  n.core = 4
)
mod_spls <- spls::splsda(
  x = as.matrix(training[, !"cancer"]), 
  y = training$cancer, 
  K = mod_splscv$K.opt, 
  eta = mod_splscv$eta.opt, 
  classifier = "logistic")
mod_spls

# predict test set
ypred <- spls::predict.splsda(
  mod_spls, 
  testing[, !"cancer"], 
  fit.type = "response"
)

# quality metrics
Metrics::logLoss(testing$cancer, ypred)
Metrics::auc(testing$cancer, ypred)

# Final model
# Train with all the samples (cv for param optim)
set.seed(4)
mod_splscv_final <- spls::cv.splsda(
  x = as.matrix(df_model[, !c("cancer", "patientid")]), 
  y = df_model$cancer, 
  K = 1:8, 
  eta = seq(from = .6, to = .95, by = .05), 
  classifier = "logistic", 
  n.core = 4
)
mod_spls_final <- spls::splsda(
  x = as.matrix(df_model[, !c("cancer", "patientid")]), 
  y = df_model$cancer, 
  K = mod_splscv_final$K.opt, 
  eta = mod_splscv_final$eta.opt, 
  classifier = "logistic")
mod_spls_final

# save results
df_eval <- df_model[!patientid %in% patients_train][,!c("cancer")]
pred_out <- spls::predict.splsda(
  mod_spls_final, 
  newx = df_eval[, !"patientid"], 
  fit.type = "response"
)
names(pred_out) <- df_eval$patientid

df_submit <- df_eval[,.(patientid)]
pat_na <- setdiff(df_submit$patientid, df_eval$patientid)
v_na <- setNames(rep(NA_prop, length(pat_na)), pat_na)

v_out <- c(pred_out, v_na)
df_out <- data.frame(
  id = names(v_out), 
  cancer = v_out, 
  stringsAsFactors = FALSE
)
write.csv(df_out, file = paste0(path_repo,"data/submissions/08_submission_lqspa.csv"), row.names = FALSE, quote = FALSE)
