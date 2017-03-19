# IMPORTS ------------------------------------------------------------------------------------------
source(paste0(path_repo,"src/be_final_ensemble/fp_model.R"))

# DATA ---------------------------------------------------------------------------------------------
## UNET Data ---------------------------------------------------------------------------------------

vars_nodules <- fread(paste0(path_repo,"data/final_model/dl_unet_v01_mingot_pc.csv"))
vars_nodules[,patientid:=gsub("dsb_","",patientid)]


vars_nodules = vars_nodules[(!x %in% extreme_values) & (!y %in% extreme_values)]  

### Filter by fp model
vars_nodules[,nodule_pred:=predictCv(fp_model, vars_nodules)]
vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.9)]


## RESNET Data -------------------------------------------------------------------------------------

vars_nodules_patches <- data.table(read.csv(paste0("D:/dsb/noduls_patches_v05_backup3.csv")))
vars_nodules_patches <- vars_nodules_patches[grep("dsb_",filename)][!is.na(x)]
vars_nodules_patches[,patientid:=gsub(".npz|dsb_","",filename)]
vars_nodules_patches[,filename := NULL]
### Filter by score
vars_nodules_patches = vars_nodules_patches[score>0.9]
names_change <- c("x","y","diameter","score")
setnames(vars_nodules_patches,names_change,paste0(names_change,"_patches"))

## Merging al nodules variables
vars_nodules <- merge(
  vars_nodules,
  vars_nodules_patches,
  all.x = T,
  all.y=T,
  by = c("nslice","patientid")
  )
## Aggregating to patient level
dataset_nodules <- aggregate_patient(vars_nodules)


## SLICES OUTPUT Data ------------------------------------------------------------------------------

dataset_slices <- fread(paste0(path_repo,"/src/dl_model_slices/output.csv"))
dataset_slices <- dataset_slices[grep("dsb_",id)]
setnames(dataset_slices,names(dataset_slices),paste0("patient_",names(dataset_slices)))
dataset_slices[,patient_id := gsub(".npz|dsb_","",patient_id)]
setnames(dataset_slices,"patient_id","patientid")


## PATIENTS AND LABELS Data ------------------------------------------------------------------------

## Add variables to all sets
annotations = fread(paste0(path_repo,"data/stage1_labels.csv"))
submission = fread(paste0(path_repo,"/data/stage1_sample_submission.csv"))
submission[,cancer := 0]

#Saving which patients are belong to training
patients_train <- annotations[,id]

# Binding train and scoring
dataset <- rbind(annotations,submission)
setnames(dataset, "id", "patientid")
dataset[,cancer := as.factor(cancer)]

## Joining all the patient variables
dataset_final <- merge(dataset, dataset_nodules, all.x=T, by="patientid")
dataset_final <- merge(dataset_final,dataset_slices,all.x = T, by = "patientid")
#dataset_final[,no_slices_with_nodules := as.numeric(is.na(total_nodules_unet))]
dataset_final <- na_to_zeros(dataset_final,names(dataset_final))


# SEPARATING TRAIN AND SCORING ---------------------------------------------------------------------

vars_train <- c(
  "max_intensity",
  "max_diameter",
  "num_slices_patches",
  "max_score",
  "max_score_patches",
  "patient_min",
  "patient_mean",
  "patient_std"
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
tr_cv = resample(lrn, train_task, rdesc, models = TRUE, measures = list(auc,logloss,fpr,fnr))
ctrlF = makeFeatSelControlGA(maxit = 2000)
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
preds <- predictCv(tr_cv,train_task)
target <- data_train[,as.numeric(as.character(cancer))]
my.AUC(target,preds)
LogLossBinary(target,preds)


# GENERATING SUBMISSION ----------------------------------------------------------------------------

preds = predictCv(final_model, scoring)

submission = data.table(id=patients_scoring, cancer=preds)
write.csv(submission, paste0(path_repo,"data/submissions/02_submission.csv"), quote=F, row.names=F)





#---------------------------------------------------------------------------------------------------



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
    .(patientid,nslice,x,y,x_patches,y_patches)]
  dt_2d_bis <- dt[
    nodule_pred > 0.2 & max_intensity > 0.96,
    .(patientid,nslice2 = nslice,x2 = x,y2 = y,x_patches2 = x_patches,y_patches2 = y_patches)]
  dt_3d <- merge(dt_2d,dt_2d_bis,all.x = T, by = "patientid",allow.cartesian = TRUE)
  dt_3d <- dt_3d[nslice2 > nslice]
  setkey(dt_3d,patientid,nslice,nslice2)
  dt_3d <- dt_3d[,.SD[1],c("patientid","nslice")]
  dt_3d <- dt_3d[nslice2-nslice < 4]
  dt_3d[,d_nodule := abs(x-x2)+abs(y-y2) < 10]
  nodules_consec <- dt_3d[,.(consec_nods = sum(d_nodule)),patientid]
  
  final_df <- merge(final_df,nodules_consec,all.x = T, by = "patientid")
  final_df[is.na(consec_nods),consec_nods := 0]
  
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