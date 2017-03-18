
source(paste0(path_repo,"src/be_final_ensemble/fp_model.R"))

# Load variables ----------------------------------------------------------

vars_nodules <- fread(paste0(path_repo,"data/final_model/dl_unet_v01_mingot_pc.csv"))
vars_nodules[,patientid:=gsub("dsb_","",patientid)]


vars_nodules = vars_nodules[(!x %in% extreme_values) & (!y %in% extreme_values)]  

# # FILTER 2: small and large nodules (1.1M -> 0.78M)
# # From cancer paper:
# # Benign nodule size (mm): mean: 4.1+-3.1, median: 3.4, quartiles: 2.7-5.0
# # Malign nodule size (mm): mean: 15.7+-12.2, median: 13, quartiles: 8-19
# # each pixel is 7mm. We accept diameter in [2.5mm, 20mm] -> [3.5, 28] pixel units of diameter
# summary(vars_nodules$diameter)
# vars_nodules = vars_nodules[diameter %between% c(3.5, 28)]


# # FILTER 3: just high intensity nodules
# vars_nodules = vars_nodules[mean_intensity>0.8]

# FILTER 4: 
vars_nodules[,nodule_pred:=predictCv(fp_model, vars_nodules)]
vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.9)]


# var_nodules_dl_patches --------------------------------------------------
# 
vars_nodules_patches = fread(paste0(path_repo,"output/noduls_patches_v04.csv"))
vars_nodules_patches = vars_nodules_patches[substr(filename,1,3)=='dsb']
vars_nodules_patches[,patientid:=gsub(".npz|dsb_","",filename)]
vars_nodules_patches[,filename:=NULL]
vars_nodules_patches = vars_nodules_patches[score>0.9]
names_change <- c("x","y","diameter","score")
setnames(vars_nodules_patches,names_change,paste0(names_change,"_patches"))
vars_nodules <- merge(vars_nodules,vars_nodules_patches,all.x = T,all.y=T,by = c("nslice","patientid"))

# Construct training ------------------------------------------------------

## Add annotations
annotations = fread(paste0(path_repo,"data/stage1_labels.csv"))
setnames(annotations, "id", "patientid")
data_train = merge(annotations, vars_nodules, by="patientid", all.x=T) #, all.y=T)
data_train = data[!is.na(cancer)]

mean(data_train[!is.na(nslice)]$cancer)
mean(data_train[is.na(nslice)]$cancer)

# Construct test ----------------------------------------------------------

submission = fread(paste0(path_repo,"/data/stage1_sample_submission.csv"))
setnames(submission, "id", "patientid")
submission = merge(submission, vars_nodules, by="patientid", all.x=T)

submission_df <- aggregate_patient(submission)
patients_sub <- submission_df[,patientid]
submision_df[,patientid := NULL]


# Table aggregation --------------------------------------------------------

# nodules per person

final_df <- aggregate_patient(data_train)
patients <- final_df[,patientid]
final_df[,patientid := NULL]
#final_df <- final_df[,.(max_intensity,max_mean_intensity,max_score,cancer)]
train_task <- makeClassifTask(data = data.frame(final_df),target = "cancer")
fv <- generateFilterValuesData(train_task, method = c("anova.test","chi.squared"))

lrn = generateModel("classif.logreg")$lrn
k = 5 #Folds
rdesc = makeResampleDesc("CV", iters = k, stratify = TRUE)

# K-fold and training
parallelStartSocket(5)
tr_cv = resample(lrn, train_task, rdesc, models = TRUE,measures = list(auc,logloss,fpr,fnr))
knitr::knit_print(tr_cv$measures.test)
final_model = train(lrn,train_task)
parallelStop()


# data visualization ------------------------------------------------------

arcvi_descriptivos = function(feature,target,equidistributed=T,bins=10){ 
  df = data.table(feature,target)
  
  if(class(feature) %in% c("numeric","integer")){
    if(uniqueN(feature)<7){
      df[,feature:=as.character(feature)]  
    }else{
      if(equidistributed) breaks = unique(quantile(feature, probs=seq(0, 1, 1.0/bins)))
      else breaks = bins # for equidistributed
      df[,feature:=cut(feature, breaks=breaks, include.lowest=T)]
    }
  }
  
  df[is.na(feature), feature:="NA"]
  df = df[,.(prob=mean(target), prct=.N/nrow(df), num=.N), by=feature][order(feature)]
  
  x = ggplot(df) +
    geom_bar(aes(x = feature, weight = prct)) +
    geom_line(aes(x = as.numeric(factor(feature)), y = prob)) + 
    theme(axis.text.x = element_text(angle = 40, hjust = 1),
          axis.title.x=element_blank(),
          axis.title.y=element_blank())
  print(x)
  
  return(df)
}

# visualize nodules distribution
library(MASS)
pid = sample(unique(data$patientid),1)
dens = kde2d(data[patientid==pid]$x, data[patientid==pid]$y)
filled.contour(dens)
xx[patientid==pid]

arcvi_descriptivos(final_df$N, final_df$cancer, equidistributed=F)
arcvi_descriptivos(final_df[max_mean_intensity>0.853]$max_mean_intensity, 
                   final_df[max_mean_intensity>0.853]$cancer, equidistributed=F)
auc(final_df$cancer, final_df$lot_nodules)  # 0.51
auc(final_df$cancer, final_df$few_nodules)  # 0.49
auc(final_df$cancer, final_df$high_intensity)  # 0.49



# training ----------------------------------------------------------------


MultiLogLoss = function(act, pred){
  eps = 1e-15
  pred = pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

k = 4 #Folds
final_df$id = sample(1:k, nrow(final_df), replace = TRUE)
list = 1:k

aucs = c()
lls = c()
for (i in 1:k){
  trainingset = subset(final_df, id %in% list[-i])
  testset = subset(final_df, id %in% c(i))
  
  # train
  #mymodel = randomForest(trainingset$target ~ ., data = trainingset, ntree = 100)
  mymodel = glm(#cancer ~ . -id, family=binomial(link='logit'), 
    #cancer ~ 1 + max_intensity + max_mean_intensity, family=binomial(link='logit'), 
    cancer ~ . , family=binomial(link='logit'), 
    data=trainingset[,!names(trainingset)%in%'patientid',with=F])
  
  # test
  pred = predict(mymodel, testset[,!names(testset)%in%'patientid',with=F], type="response")
  #pred = rep(.25, length(pred))
  real = testset$cancer
  
  # store results
  aucs = c(aucs, auc(real,pred))
  lls = c(lls, MultiLogLoss(real,pred))
}
summary(aucs)
summary(lls)


# final model (all data)
final_model = glm(cancer ~ 1 + max_intensity + max_mean_intensity, family=binomial(link='logit'), 
                  data=final_df[,!names(testset)%in%'patientid',with=F])
summary(final_model)


# submission --------------------------------------------------------------


preds = predictCv(final_model, submission_df)

submission = data.table(id=patients_sub, cancer=preds)
write.csv(submission, paste0(path_repo,"data/submissions/be00_submission.csv"), quote=F, row.names=F)



aggregate_patient <- function(dt) {
  final_df = dt[,.(total_nodules_unet=sum(!is.na(x)), 
                     total_nodules_patches = sum(!is.na(x_patches)),
                     num_slices_unet=uniqueN(nslice[!is.na(x)]),
                     num_slices_patches=uniqueN(nslice[!is.na(x_patches)]),
                     nodules_per_slice_unet=sum(!is.na(x))/uniqueN(nslice[!is.na(x)]),
                     nodules_per_slice_patches=sum(!is.na(x_patches))/uniqueN(nslice[!is.na(x_patches)]),
                     max_intensity=max(max_intensity, na.rm=T), 
                     max_mean_intensity=max(mean_intensity, na.rm=T),
                     min_intensity = min(min_intensity,na.rm=T),
                     max_score=max(nodule_pred, na.rm=T),
                     mean_score=mean(nodule_pred, na.rm=T)
  ),
  by=.(patientid,cancer)]
  
  final_df[!is.finite(max_intensity), max_intensity:=0]
  final_df[!is.finite(min_intensity), min_intensity:=0]
  final_df[!is.finite(max_mean_intensity), max_mean_intensity:=0]
  final_df[is.na(final_df)] <- 0
  final_df[max_score < 0,max_score := 0]
  return(final_df)
}