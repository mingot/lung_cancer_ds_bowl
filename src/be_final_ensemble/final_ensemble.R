
## REPO PATH, CANVIAR SI ES NECESSARI
path_repo <<- "D:/lung_cancer_ds_bowl/"
path_data <<- "D:/output/"
# IMPORTS ------------------------------------------------------------------------------------------
source(paste0(path_repo,"src/be_final_ensemble/config.R"))
source(paste0(path_repo,"src/be_final_ensemble/fp_model.R"))

# DATA ---------------------------------------------------------------------------------------------

## PATIENTS AND LABELS Data ------------------------------------------------------------------------

## Add variables to all sets
annotations = fread(paste0(path_repo,"data/stage1_labels.csv"))
submission = fread(paste0(path_repo,"/data/stage1_sample_submission.csv"))
submission[,cancer := 0]

#Saving which patients are belong to training
patients_train <- annotations[,id]

# Binding train and scoring
patients <- rbind(annotations,submission)
setnames(patients, "id", "patientid")
patients[,cancer := as.factor(cancer)]


## UNET Data ---------------------------------------------------------------------------------------

vars_nodules <- fread(paste0(path_repo,"data/final_model/dl_unet_v01_mingot_pc.csv"))
vars_nodules[,patientid:=gsub("dsb_","",patientid)]
vars_nodules = vars_nodules[(!x %in% extreme_values) & (!y %in% extreme_values)]  

### Filter by fp model
vars_nodules[,nodule_pred:=predictCv(fp_model, vars_nodules)]
vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.9)]
# Merging with patients to get cancer info for easier debug
vars_nodules <- merge(vars_nodules,patients,all.x=T,by = "patientid")


## RESNET Data -------------------------------------------------------------------------------------
#vars_nodules_patches <- fread(paste0("D:/dsb/nodules_patches_v05_augmented.csv"))
vars_nodules_patches <- fread(paste0("D:/dsb/noduls_patches_v06_rectif.csv")) ## PATH
vars_nodules_patches <- vars_nodules_patches[grep("dsb_",patientid)][!is.na(x)]
vars_nodules_patches[,patientid:=gsub(".npz|dsb_","",patientid)]
### Filter by score
vars_nodules_patches = vars_nodules_patches[score>0.9]
names_change <- c("x","y","diameter","score")
setnames(vars_nodules_patches,names_change,paste0(names_change,"_patches"))
vars_nodules_patches <- merge(vars_nodules_patches,patients,all.x=T,by = "patientid")

## Merging al nodules variables
vars_nodules <- rbind(vars_nodules,vars_nodules_patches,fill = TRUE)
vars_nodules[,cancer := NULL]
## Aggregating to patient level
dataset_nodules <- aggregate_patient(vars_nodules)

## SLICES OUTPUT Data ------------------------------------------------------------------------------

dataset_slices <- fread(paste0(path_repo,"/src/dl_model_slices/output.csv"))
dataset_slices <- dataset_slices[grep("dsb_",id)]
setnames(dataset_slices,names(dataset_slices),paste0("patient_",names(dataset_slices)))
dataset_slices[,patient_id := gsub(".npz|dsb_","",patient_id)]
setnames(dataset_slices,"patient_id","patientid")


## Joining all the patient variables
dataset_final <- merge(patients,dataset_nodules,all.x = T, by = "patientid")
dataset_final <- merge(dataset_final,dataset_slices,all.x = T, by = "patientid")
dataset_final <- na_to_zeros(dataset_final,names(dataset_final))

# SEPARATING TRAIN AND SCORING ---------------------------------------------------------------------

vars_train <- c(
  #"max_intensity",
  "max_diameter",
  "big_nodules_patches",
  "max_diameter_patches",
  "num_slices_patches",
  "max_score",
  "max_score_patches",
  "nslice_nodule_patch",
  "consec_nods_patches",
  "diameter_nodule_patch"
  #"patient_min",
  #"patient_mean",
  #"patient_std"
  #"diameter_nodule"
  #"max_intensity_nodule",
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

lrn = generateModel("classif.gbm")$lrn
k_folds = 5
rdesc = makeResampleDesc("CV", iters = k_folds, stratify = TRUE)


# K-FOLD METRICS AND TRAINING MODEL ----------------------------------------------------------------

parallelStartSocket(5)
tr_cv = resample(lrn, train_task, rdesc, models = TRUE, measures = list(auc,logloss,fpr,fnr))
ctrlF = makeFeatSelControlGA(maxit = 3000)
# sfeats = selectFeatures(
#   learner = lrn,
#   task = train_task,
#   resampling = rdesc,
#   control = ctrlF,
#   measures = logloss,
#   show.info = FALSE)
knitr::knit_print(tr_cv$measures.test)
final_model = train(lrn,train_task)
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
write.csv(submission, paste0(path_repo,"data/submissions/04_submission.csv"), quote=F, row.names=F)





#---------------------------------------------------------------------------------------------------

