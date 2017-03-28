
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
features_sp <- fread(paste0(path_repo,"src/sp_final_ensemble/submissions/sp_01_features.csv"))
dataset_final <- merge(dataset_final,features_sp,all.x = T,by = "patientid")
dataset_final <- na_to_zeros(dataset_final,names(dataset_final))


vars_train <- c(
  "max_intensity",
  "max_diameter",
  "big_nodules_patches",
  "max_diameter_patches",
  "num_slices_patches",
  "max_score",
  "max_score_patches",
  "nslice_nodule_patch",
  "consec_nods_patches",
  "diameter_nodule_patch",
  "score_2_patch",
  "max_score_2"
  #"score_mean",
  #"nslice_sd",
  #"diameter_sd"
  #"patient_max",
  #"patient_min",
  #"patient_mean",
  #"patient_std"
  #"diameter_nodule"
  #"max_intensity_nodule"
  #"mean_intensity_nodule"
  )
#vars_train <- names(dataset_final)
dataset_final_f <- dataset_final[,.SD,.SDcols = unique(c(vars_train,"patientid","cancer"))]
data_train <- dataset_final_f[patientid %in% patients_train]
scoring <- dataset_final_f[!patientid %in% patients_train]
patients_scoring <- scoring[,patientid]
data_train[,patientid := NULL]
scoring[,patientid := NULL]


# CREATING TRAINING TASK AND MODEL -----------------------------------------------------------------

train_task <- makeClassifTask(data = data.frame(data_train),target = "cancer")

fv <- generateFilterValuesData(train_task, method = c("anova.test","chi.squared"))
data.table(fv$data)

lrn = generateModel("classif.logreg")$lrn
k_folds = 5
rdesc = makeResampleDesc("CV", iters = k_folds, stratify = TRUE)


# K-FOLD METRICS AND TRAINING MODEL ----------------------------------------------------------------

parallelStartSocket(5)
set.seed(123)

tr_cv = resample(lrn, train_task, rdesc, models = TRUE, measures = list(auc,logloss,fpr,fnr))
knitr::knit_print(tr_cv$measures.test)
summary(tr_cv$measures.test$auc)
summary(tr_cv$measures.test$logloss)

# ctrlF = makeFeatSelControlGA(maxit = 4000)
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


# GENERATING SUBMISSION ----------------------------------------------------------------------------

preds = predictCv(final_model, scoring)

submission = data.table(id=patients_scoring, cancer=preds)
write.csv(submission, paste0(path_repo,"data/submissions/08_submission.csv"), quote=F, row.names=F)


# GENERATING PREDICTIONS FOR TRAINING ----------------------------------------------------------------------------

preds = predictCv(final_model, train_task)
data_out <- copy(data_train)
data_out$patientid = patients_train
data_out$predicted=preds
write.csv(data_out, paste0(path_repo,"data/final_model/scoring_train.csv"), quote=F, row.names=F)


#---------------------------------------------------------------------------------------------------

