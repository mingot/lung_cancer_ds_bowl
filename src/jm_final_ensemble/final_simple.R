library(xgboost)
library(data.table)
library(pROC)
library(dplyr)
library(ggplot2)

# INITIAL SETUP -----------------------------------------------------------------

path_repo <<- "~/lung_cancer_ds_bowl/"
path_dsb <<- "/home/shared/output/"
annotations = fread(paste0(path_repo,"/data/stage1_labels.csv"))
submission = fread(paste0(path_repo,"/data/stage1_sample_submission.csv"))
submission[,cancer := 0]
submission[,dataset := "scoring"]
annotations[,dataset := "training"]

#Saving which patients are belong to training

# Binding train and scoring
patients <- rbind(annotations,submission)
setnames(patients, "id", "patientid")
patients[,patientid:=paste0("dsb_",patientid,".npz")]


# FEATURES -----------------------------------------------------------------

dl1 = fread("/home/shared/output/resnet/nodules_patches_dl1_v11.csv")
dl2 = fread("/home/shared/output/resnet/nodules_patches_hardnegative_v03.csv")

dl12 = merge(dl1, dl2, by=c('patientid','nslice','x','y','diameter'))
dl12[,score:=(score.x+score.y)/2]
dl12[,score.x:=NULL]
dl12[,score.y:=NULL]
# write.csv(dl12, '/home/mingot/tmp/input_aggregation.csv', row.names=F, quote=F)
# python mergeNodules.py -t 0.5 ~/tmp/input_aggregation.csv ~/tmp/output_aggregation.csv

dl12 = dl12[score>0.7]
dl12_final = dl12[,.(total_nodules_patches = sum(!is.na(x)),
                     big_nodules_patches = sum(diameter > 20,na.rm = T),
                     max_diameter_patches = max(diameter,na.rm = T),
                     num_slices_patches = uniqueN(nslice[!is.na(x)]),
                     nodules_per_slice_patches = sum(!is.na(x))/uniqueN(nslice[!is.na(x)]),
                     max_score_patches = max(score,na.rm=T),
                     mean_score_patches = mean(score,na.rm=T)),by=patientid]

dl12[,`:=`(
  max_score = max(nodule_pred,na.rm=T),
  max_score_patches=max(score_patches,na.rm=T)),patientid]

max_score_nodule = dl12[max_score_patches == score_patches & max_score_patches > 0,.(patientid,
                                                                                     nslice_nodule_patch = nslice,
                                                                                     diameter_nodule_patch = diameter_patches)]


# NODULES AGGREGATED
nagg = fread("/home/mingot/tmp/output_aggregation.csv")
nagg = nagg[nslicesSpread > 1]
nagg[,`:=`(max_score_filtered = max(score),max_nslicesSpread = max(nslicesSpread)), by = patientid]
nagg1 = nagg[, .(max_nsliceSpread = max(nslicesSpread),
                  max_score_filtered = max(score),
                  max_diameter_filtered = max(diameter),
                  n_nodules_filtered = .N), patientid]
nagg2 = nagg[max_score_filtered == score,.SD[1],patientid][,.(patientid,
                                                              diameter_score_filtered = diameter,
                                                              nslice_score_filtered = nslice,
                                                              nsliceSpread_max = nslicesSpread,
                                                              x_score_filtered = x,
                                                              y_score_filtered = y)]
nagg3 = nagg[max_nslicesSpread == nslicesSpread,.SD[1],patientid][,.(patientid,
                                                                     max_score_spread = score,
                                                                     diameter_slices_filtered = diameter,
                                                                     nslice_slices_filtered = nslice,
                                                                     x_score_spread = x,
                                                                     y_score_spread = y)]
nagg_final = merge(nagg1, nagg2, by='patientid')
nagg_final = merge(nagg_final, nagg3, by='patientid')

# EMPHYSEMA
emphysema = fread(paste0(path_dsb,"emphysema/var_emphysema_v07.csv"))
setnames(emphysema,names(emphysema),c("patientid","emph_var1","emph_var2","emph_var3","emph_var4","emph_var5"))
emphysema[,patientid:=paste0("dsb_",patientid,".npz")]

# EXTRA FEAT
extra_feats <- fread(paste0(path_dsb,"extrafeatures/stage1_extra_features_intercostal.csv"))
setnames(extra_feats,names(extra_feats),c("patientid", 'extraf_nperc_std', 'extraf_nperc_avg', 'extraf_bone_density', 'extraf_nperc_mode'))


# FEATURES augment
features_sp = fread(paste0(path_dsb,"/sp_04_features.csv"))
features_sp[,patientid:=paste0("dsb_",patientid,".npz")]
df_nodule = fread('/home/shared/output/resnet/nodules_patches_dl1_v11_score07_noduleaggr_augmented.csv')
df_nodule <- plyr::ddply(
  df_nodule, 
  "patientid", 
  function(df)  {
    # browser()
    df_top <- dplyr::arrange(df, desc(score)) %>% head(1) %>% as.data.table
    df_out <- df_top[, !"patientid"] #%>% summarise_each(funs(min, mean, max, sd))
    df_out$man_nslicespread_sd <- sd(df$nslicesSpread)
    df_out
  }, .progress = "text"
) %>% as.data.table
#df_nodule = df_nodule[,.(patientid,PC1_lbp, man_nslicespread_sd)]


## consecutive nodules
dt_2d <- dl12[!is.na(x),.(patientid,nslice,x,y)]
dt_2d_bis <- dl12[!is.na(x),.(patientid,nslice2 = nslice,x2=x,y2=y)]
dt_3d <- merge(dt_2d,dt_2d_bis,all.x=T,by="patientid",allow.cartesian = TRUE)
dt_3d <- dt_3d[nslice2 > nslice]
setkey(dt_3d,patientid,nslice,nslice2)
dt_3d <- dt_3d[nslice2-nslice <3]
dt_3d[,d_nodule := abs(x-x2) + abs(y-y2) < 10]
dt_3d <- dt_3d[,.SD[1],c("patientid","nslice")]
nodules_patches_consec <- dt_3d[,.(consec_nods_patches = sum(d_nodule)),patientid]


## FINAL
final_dt = merge(patients, nagg_final, by='patientid', all.x=T)
final_dt = merge(final_dt, dl12_final, by='patientid', all.x=T)
final_dt = merge(final_dt, emphysema, by='patientid', all.x=T)
final_dt = merge(final_dt, extra_feats, by='patientid', all.x=T)
final_dt = merge(final_dt, features_sp, by='patientid', all.x=T)
final_dt = merge(final_dt, df_nodule, by='patientid', all.x=T)
final_dt = merge(final_dt, nodules_patches_consec, by='patientid', all.x=T)
final_dt[is.na(final_dt)] = 0


# vessels_marti -----------------------------------------------------------

vs_df = fread('/home/shared/output/vessels_marti.csv')
vs_df = vs_df[!is.na(vessel_sum)]
vs_df = vs_df[,.(vs_num=.N),by=patientid]

# model -------------------------------------------------------------------

MultiLogLoss = function(act, pred){
  eps = 1e-15
  pred = pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}


data_train = final_dt[dataset=='training']
vars_sel = names(data_train)
vars_sel = vars_sel[!vars_sel%in%c('patientid','dataset','id','cancer')]
#vars_sel = c("max_score_patches","max_diameter_patches","big_nodules_patches","diameter_median","y_score_filtered","max_nsliceSpread","score_sd","03_area_max")
vars_sel =c('max_diameter_patches','max_nsliceSpread','max_score_patches','consec_nods_patches','nsliceSpread_max','diameter_slices_filtered','PC1_lbp','man_nslicespread_sd')
vars_sel =c('max_diameter_patches','max_score_patches','PC1_lbp','man_nslicespread_sd')
#vars_sel <- c("max_diameter_patches","max_nsliceSpread","max_score_patches","consec_nods_patches","nsliceSpread_max","diameter_slices_filtered") # nslice_nodule_patch

aucs = c(); lls = c()
for(i in 1:20){
k = 3; data_train$id = sample(1:k, nrow(data_train), replace = TRUE); list = 1:k
for (i in 1:k){
  trainingset = subset(data_train, id %in% list[-i])
  testset = subset(data_train, id %in% c(i))
  
  ##train
  mymodel = glm(cancer ~ 1 + ., family=binomial(link='logit'),
                data=trainingset[,c("cancer",vars_sel),with=F])

  # mymodel = xgboost(data = as.matrix(trainingset[,vars_sel,with=F]), label = trainingset$cancer,
  #                         max.depth = 2, eta = 1, nthread = 2, nround = 2,
  #                         lambda = 2,
  #                         objective = "binary:logistic", eval_metric="logloss")
  
  # test
  #pred = predict(mymodel, as.matrix(testset[,vars_sel,with=F]))#, n.trees=100)
  pred = predict(mymodel, testset[,c("cancer",vars_sel),with=F], type="response")#, n.trees=100)
  real = testset$cancer
  
  # store results
  aucs = c(aucs, auc(real,pred))
  lls = c(lls, MultiLogLoss(real,pred))
}
}
summary(mymodel); summary(aucs); summary(lls)
importance_matrix <- xgb.importance(model = mymodel)
print(importance_matrix)
vars_sel[as.numeric(importance_matrix$Feature)]

# other: "total_nodules_patches","n_patches","big_nodules_patches","max_nsliceSpread","06_mean_intensity","extraf_bone_density","y_score_spread","03_area"

# step to select features
library(MASS)
fit <- lm(cancer~., data=data_train[,c('cancer',vars_sel),with=F])
step <- stepAIC(fit, direction="backward")
step$anova # display results




# visualization -----------------------------------------------------------

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

arcvi_descriptivos(data_train$total_nodules_patches, data_train$cancer)
arcvi_descriptivos(data_train$big_nodules_patches, data_train$cancer)
arcvi_descriptivos(data_train$`06_mean_intensity`, data_train$cancer)  # TO REVIEW!
arcvi_descriptivos(data_train$y_score_spread, data_train$cancer) # TO REVIEW!
arcvi_descriptivos(data_train$`03_area`, data_train$cancer)  # TO REVIEW
arcvi_descriptivos(data_train$PC1_lbp, data_train$cancer) 
arcvi_descriptivos(data_train$`10_lungmask`, data_train$cancer) 
arcvi_descriptivos(data_train$vs_num, data_train$cancer) 
arcvi_descriptivos(data_train$`40_nodeverticalposition`, data_train$cancer) 

arcvi_descriptivos(data_train$emph_var1, data_train$cancer, equidistributed=F) 
arcvi_descriptivos(data_train$emph_var2, data_train$cancer, equidistributed=F) 
arcvi_descriptivos(data_train$emph_var3, data_train$cancer, equidistributed=F) 
arcvi_descriptivos(data_train$emph_var4, data_train$cancer, equidistributed=F) 
arcvi_descriptivos(data_train$emph_var5, data_train$cancer, equidistributed=F) 

data_train[,feat1:=as.numeric(`06_mean_intensity`>0.75)]
data_train[,feat2:=as.numeric(124<y_score_spread & y_score_spread<200)]
data_train[,feat3:=as.numeric(`03_area`>314)]



