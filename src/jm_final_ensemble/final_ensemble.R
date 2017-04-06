
# install.packages("data.table")
# install.packages("mlr")
# install.packages("knitr")
# install.packages("Rfast")
# install.packages("FSelector")
# install.packages("gbm")
# install.packages("pROC")
# install.packages("xgboost")


## REPO PATH
path_repo <<- "~/lung_cancer_ds_bowl/"
path_dsb <<- "/home/shared/output/"

# IMPORTS ------------------------------------------------------------------------------------------
source(paste0(path_repo,"src/jm_final_ensemble/config.R"))
source(paste0(path_repo,"src/jm_final_ensemble/fp_model.R"))
source(paste0(path_repo,"src/jm_final_ensemble/aggregate_dt.R"))

## PATIENTS AND LABELS Data ------------------------------------------------------------------------
## Add variables to all sets

dataset_final <- generate_patient_dt(path_repo,path_dsb)


# SEPARATING TRAIN AND SCORING ---------------------------------------------------------------------
patients_train <- dataset_final[dataset == "training",patientid]
dataset_final[,dataset := NULL]
features_sp <- fread(paste0(path_dsb,"/sp_04_features.csv"))
dataset_final <- merge(dataset_final,features_sp,all.x = T,by = "patientid")
dataset_final <- na_to_zeros(dataset_final,names(dataset_final))
nombres_m <- names(dataset_final)
for(n in nombres_m) {
  if(length(unique(dataset_final[[n]])) <= 1) dataset_final[[n]] <- NULL
}


vars_train <- c(
  #"max_intensity",
  # "max_diameter",
  #"big_nodules_patches",
  # "nods_15",
  # "nods_20",
  # "nods_25",
  #"nods_30",
  "max_diameter_patches",
  #"max_score_filtered",
  "max_nsliceSpread",
  #"max_diameter_filtered",
  #"num_slices_patches",
  #"max_score",
  "max_score_patches",
  "nslice_nodule_patch",
  "consec_nods_patches",
  #"diameter_nodule_patch",
  #"n_nodules_filtered",
  #"max_score_filtered",
  #"diameter_score_filtered",
  "nsliceSpread_max",
  #"max_score_spread",
  "diameter_slices_filtered"
  #"score_mean",
  #"nslice_sd",
  #"diameter_sd"
  #"patient_max",
  #"patient_min",
  #"patient_mean",
  #"patient_std"
  #"diameter_nodule"
  # "max_intensity_nodule",
  # "mean_intensity_nodule"
  )
vars_sp <- c(
  #"PC3_lbp_min",
  #"score_median",
  #"diameter_sd"
  #"PC1_lbp_sd"
)
vars_train <- c(vars_train,vars_sp)
vars_train <- names(dataset_final)
print(vars_train[!vars_train %in% names(dataset_final)])
dataset_final_f <- dataset_final[,.SD,.SDcols = unique(c(vars_train,"patientid","cancer"))]
data_train <- dataset_final_f[patientid %in% patients_train]
scoring <- dataset_final_f[!patientid %in% patients_train]
patients_scoring <- scoring[,patientid]
data_train[,patientid := NULL]
scoring[,patientid := NULL]


# CREATING TRAINING TASK AND MODEL -----------------------------------------------------------------

train_task <- makeClassifTask(data = data.frame(data_train),target = "cancer")

fv <- generateFilterValuesData(train_task, method = c("anova.test","chi.squared"))
vars_importance <- data.table(fv$data)
vars_importance[chi.squared > 0]

lrn = generateModel("classif.gbm")$lrn
#lrn = generateModel("classif.logreg")$lrn
#params = generateModel("classif.")$ps
k_folds = 5
rdesc = makeResampleDesc("CV", iters = k_folds, stratify = TRUE)


# K-FOLD METRICS AND TRAINING MODEL ----------------------------------------------------------------

parallelStartSocket(5)
set.seed(123)

tr_cv = resample(lrn, train_task, rdesc, models = TRUE, measures = list(auc,logloss,fpr,fnr))
knitr::knit_print(tr_cv$measures.test)
summary(tr_cv$measures.test$auc)
summary(tr_cv$measures.test$logloss)
# ctrlT = makeTuneControlGenSA(maxit = 10)
ctrlF = makeFeatSelControlGA(maxit = 4000)
# res = tuneParams(
#   learner = lrn,
#   task = train_task,
#   resampling = rdesc,
#   control = ctrlT,
#   measures = logloss,
#   par.set = params
# )
# sfeats = selectFeatures(
#   learner = lrn,
#   task = train_task,
#   resampling = rdesc,
#   control = ctrlF,
#   measures = logloss,
#   show.info = FALSE)

final_model = mlr::train(lrn,train_task)
summary(final_model$learner.model)
parallelStop()

# train_metrics
preds <- predictCv(final_model,train_task)
target <- data_train[,as.numeric(as.character(cancer))]
my.AUC(target,preds)
LogLossBinary(target,preds)



# DL3 features ------------------------------------------------------------

# validation patients
validation = fread(paste0(path_repo, 'data/stage1_validation.csv'))
validation[,patientid:=gsub(".npz|dsb_","",patientid)]

# Contruct partial training set (over 993 patients)
data_train_partial = dataset_final_f[(!patientid %in% validation$patientid) & (patientid %in% patients_train)]  # patients from train and validation sets
data_train_partial[,patientid:=NULL]
train_task_partial <- makeClassifTask(data = data.frame(data_train_partial),target = "cancer")
partial_model = mlr::train(lrn,train_task_partial)

# Execute
data_test = dataset_final_f[patientid %in% validation$patientid]
preds = predictCv(partial_model,data_test)
submission_partial = data.table(patientid=data_test$patientid, preds=preds, cancer=data_test$cancer)
write.csv(submission_partial, paste0(path_dsb,"dl3_train/partial_submission_16.csv"), quote=F, row.names=F)

preds = predictCv(final_model, scoring)
submission_total = data.table(patientid=patients_scoring, preds=preds)
write.csv(submission_total, paste0(path_dsb,"dl3_train/total_submission_16.csv"), quote=F, row.names=F)

# GENERATING SUBMISSION ----------------------------------------------------------------------------

preds = predictCv(final_model, scoring)

submission = data.table(id=patients_scoring, cancer=preds)
mean(submission$cancer)
write.csv(submission, paste0(path_repo,"data/submissions/20_submission.csv"), quote=F, row.names=F)


# GENERATING PREDICTIONS FOR TRAINING ----------------------------------------------------------------------------

preds = predictCv(final_model, train_task)
data_out <- copy(data_train)
data_out$patientid = patients_train
data_out$predicted=preds
write.csv(data_out, paste0(path_repo,"data/final_model/scoring_train_20.csv"), quote=F, row.names=F)


#---------------------------------------------------------------------------------------------------

