library(data.table)
library(ggplot2)
library(pROC)
library(xgboost)


# Load variables ----------------------------------------------------------

vars_nodules = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/dl_unet_v01_mingot_pc.csv'))
vars_nodules[,patientid:=gsub("dsb_","",patientid)]
# hog = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/hog_v1.csv', header=F))
# setnames(hog, c('V1','V2','V3','V4','V5'), c('score','patientid','nslice','x','y'))
# hog[,patientid:=gsub("luna_","",patientid)]
# hog[,patientid:=gsub(".npz","",patientid)]

# luna_nodules = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/annotations.csv'))
# summary(luna_nodules$diameter_mm)
# quantile(luna_nodules$diameter_mm,0.9)

# FILTER 1: extreme windows (1.5M -> 1.1M)
vars_nodules = vars_nodules[x!=0 & x!=511 & y!=0 & y!=511]  

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
vars_nodules[,nodule_pred:=predict(fp_model, data, type="response")]
vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.9)]


# v11 vs v18
# rows: 829k vs 29k
# score: same histograms, different levels
#  v18 - TH 0.6 -> 0.655 AUC [PROBLEMA AMB EL DIAMETRE!!]
#  v11 - TH 0.8 -> 0.663 AUC [mean_diam, max_diam, max_score]


# var_nodules_dl_patches --------------------------------------------------

vars_nodules_patches = fread("/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/dl_nodules/nodules_patches_dl1_v11.csv")
vars_nodules_patches = fread("/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/dl_nodules/nodules_patches_dl1_v18.csv")
vars_nodules_patches = vars_nodules_patches[score>0.6]
vars_nodules_patches = vars_nodules_patches[substr(patientid,1,3)=='dsb']
vars_nodules_patches[,patientid:=gsub(".npz|dsb_","",patientid)]

v11 = vars_nodules_patches
v18 = vars_nodules_patches
v18
hist(v11$score)
hist(v18$score)

vars_nodules_patches[patientid==sample(patientid,1)]

# FILTER: reduce FP with fp model
# vars_nodules[,nodule_pred:=predict(fp_model, vars_nodules, type="response")]
vars_nodules = vars_nodules_patches
# vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.98)]
uniqueN(vars_nodules$patientid)



# combined ----------------------------------------------------------------

vars_nodules_patches = mm[substr(patientid,1,3)=='dsb']
vars_nodules_patches[,patientid:=gsub(".npz|dsb_","",patientid)]
vars_nodules_patches[,score:=score_dl1]
vars_nodules_patches[,score:=(0.7*score_dl1 + 0.3*score_dl2)]

vars_nodules = vars_nodules_patches[score>0.9]
# vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.98)]
uniqueN(vars_nodules$patientid)



# Construct training ------------------------------------------------------

## Add annotations
annotations = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/stage1_labels.csv'))
setnames(annotations, "id", "patientid")
data = merge(annotations, vars_nodules, by="patientid", all.x=T) #, all.y=T)
data = data[!is.na(cancer)]

mean(data[!is.na(nslice)]$cancer)
mean(data[is.na(nslice)]$cancer)



# Construct test ----------------------------------------------------------

submission = data.table(read.csv("/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/stage1_sample_submission.csv"))
setnames(submission, "id", "patientid")
submission = merge(submission, vars_nodules, by="patientid", all.x=T)
data = submission



final_df = data[,.(total_nodules=.N,
                   nodules_per_slice=.N/uniqueN(nslice),
                   num_slices = uniqueN(nslice),
                   nodules_upper = sum(nslice>60),
                   max_diameter = max(diameter),
                   max_score = max(score),
                   mean_score = mean(score)), by=.(patientid, cancer)]

# Table aggregation --------------------------------------------------------

# nodules per person
final_df = data[,.(total_nodules = sum(!is.na(nslice)), 
                   num_slices = uniqueN(nslice[!is.na(nslice)]), 
                   nodules_per_slice = sum(!is.na(nslice))/uniqueN(nslice),
                   max_diameter = max(diameter),
                   mean_diameter = mean(diameter),
                   median_diameter = median(diameter),
                   max_score1 = max(score)
                   #max_intensity=max(max_intensity, na.rm=T), 
                   #max_mean_intensity=max(mean_intensity, na.rm=T)
                   #max_score=max(score, na.rm=T),
                   #mean_score=mean(score, na.rm=T)
                   ),
          by=.(patientid,cancer)]

final_df[,has_nodule:=as.numeric(total_nodules>0)]
final_df[,has_nodule3:=as.numeric(total_nodules>3)]

final_df[!is.finite(max_intensity), max_intensity:=0]
final_df[!is.finite(max_mean_intensity), max_mean_intensity:=0]
final_df[,high_intensity:=ifelse(max_mean_intensity>0.9,1,0)]
final_df[,lot_nodules:=ifelse(total_nodules>150,1,0)]
final_df[,few_nodules:=ifelse(total_nodules<50,1,0)]
final_df[!is.finite(max_diameter), max_diameter:=0]
final_df[!is.finite(mean_diameter), mean_diameter:=0]
final_df[!is.finite(median_diameter), median_diameter:=0]


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


arcvi_descriptivos(final_df$total_nodules, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$max_score, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$num_slices, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$max_diameter, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$mean_diameter, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$median_diameter, final_df$cancer, equidistributed=T)

arcvi_descriptivos(final_df$total_nodules, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$has_nodule, final_df$cancer, equidistributed=T)
arcvi_descriptivos(final_df$has_nodule3, final_df$cancer, equidistributed=T)

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
precs = c()
accs = c()
for (i in 1:k){
  trainingset = subset(final_df, id %in% list[-i])
  testset = subset(final_df, id %in% c(i))
  
  # train
  mymodel = glm(cancer ~ . -id, family=binomial(link='logit'), 
                #cancer ~ 1 + max_diameter + max_score2 + max_score_mean, family=binomial(link='logit'), 
                #cancer ~ 1 + total_nodules + num_slices + max_diameter, family=binomial(link='logit'), 
                data=trainingset[,!names(trainingset)%in%'patientid',with=F])
  
  # test
  pred = predict(mymodel, testset[,!names(testset)%in%'patientid',with=F], type="response")
  #pred = rep(.25, length(pred))
  pred[!is.finite(pred)] = 0.25 #0.25
  #pred = runif(length(real), min=0, max=1)
  real = testset$cancer
  
  TH = 0.5
  tp = sum(pred>TH & real==1)
  fp = sum(pred>TH & real==0)
  tn = sum(pred<TH & real==0)
  
  precs = c(precs, tp/(tp+fp))
  accs = c(accs, (tp + tn)/length(real))
  
  # store results
  aucs = c(aucs, auc(real,pred))
  lls = c(lls, MultiLogLoss(real,pred))
}
summary(aucs)
summary(lls)
summary(precs)
summary(accs)

summary(mymodel)



xgb_imp = xgb.importance( names(trainingset[,!names(final_df)%in%c('patientid','id'),with=F]), model = model )
xgb_imp


# final model (all data)
final_model = glm(cancer ~ 1 + max_intensity + max_mean_intensity, family=binomial(link='logit'), 
                  data=final_df[,!names(testset)%in%'patientid',with=F])
summary(final_model)


# submission --------------------------------------------------------------


preds = predict(final_model, final_df[,!names(testset)%in%'patientid',with=F], type="response")

submission = data.table(id=final_df$patientid, cancer=preds)
write.csv(submission, '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/submissions/00_submission.csv', quote=F, row.names=F)

