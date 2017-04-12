library(data.table)
library(pROC)
library(dplyr)
library(LiblineaR)

# INITIAL SETUP -----------------------------------------------------------------
rm(list=ls())

MultiLogLoss = function(act, pred){
  eps = 1e-15
  eps = 0
  pred = pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

# global paths
# source("/Users/mingot/Projectes/kaggle/ds_bowl_lung/src/jm_final_ensemble/config.R")

# # file locations
path_repo <<- "~/lung_cancer_ds_bowl/"
path_dsb <<- "/home/shared/output/"

# STAGE 1
ANNOTATIONS_FILE = paste0(path_repo,"data/stage1_labels.csv")  
SUBMISSIONS_FILE = paste0(path_repo,"data/stage1_sample_submission.csv")
DL3_VALIDATION_FILE = paste0(path_repo, 'data/stage1_validation.csv')

# STAGE 1 + info stage 2
ANNOTATIONS_FILE = paste0(path_repo,"data/stage1_labels_total.csv")  # STAGE1 + TESTSET RELEASED
SUBMISSIONS_FILE = paste0(path_repo,"/data/stage2_sample_submission.csv")  # STAGE 2 SUBMISSION
DL3_VALIDATION_FILE = paste0(path_repo, 'data/stage1_validation_total.csv')

# DL3
DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v06_TH04.csv")
#DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v02.csv")
#DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v06.csv")
#DL4_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v10.csv")
#DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v06_TH05.csv")
#DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v06_TH03.csv")
#DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v06_TH01.csv")

# ## DL12 V11 STAGE 1 (AUC MODEL 01: ~0.771)
# DL1_FILE = paste0(path_dsb,"resnet/nodules_patches_dl1_v11.csv")
# DL2_FILE = paste0(path_dsb,"resnet/nodules_patches_hardnegative_v03.csv")
# #NODULES_EXTENDED_FILE = paste0(path_dsb,"resnet/nodules_patches_dl1_v11_score07_noduleaggr_augmented.csv")
# NODULES_EXTENDED_FILE = paste0(path_dsb,"resnet/TEST_dl1_v11_augmented.csv")

## DL12 V11 STAGE 1 + 2 (AUC MODEL 01: ~0.776)
DL1_FILE = '/home/shared/output/resnet/v11/nodules_patches_dl1_v11_stage2_total.csv'
DL2_FILE = '/home/shared/output/resnet/v11/nodules_patches_dl2_v03_stage2_total.csv'
NODULES_EXTENDED_FILE = '/home/shared/output/resnet/v11/dl1_v11_augmented.csv' # DL1
#NODULES_EXTENDED_FILE = '/home/shared/output/resnet/v11/dl1_v11_augmented_TEST_SERGI.csv' # DL1

# ## DL12 V19 STAGE 1 + 2 (AUC MODEL 01: ~0.717)
# DL1_FILE = '/home/shared/output/resnet/v19/nodules_patches_dl1_v19_total.csv'
# DL2_FILE = '/home/shared/output/resnet/v19/nodules_patches_dl2_v04_total.csv'
# # #NODULES_EXTENDED_FILE = '/home/shared/output/resnet/v19/dl12_v19_augmented.csv' # DL1 + DL2
# NODULES_EXTENDED_FILE = '/home/shared/output/resnet/v19/dl1_v19_augmented.csv' # DL1


# # For automation
# FINAL_MODEL_01 = paste0(path_repo,"models/final_model_01.rda")
# FINAL_MODEL_02 = paste0(path_repo,"models/final_model_02.rda")
# SUBMISSION_OUTPUT01 = paste0(path_dsb, 'submissions/final_submission_01.csv')
# SUBMISSION_OUTPUT02 = paste0(path_dsb, 'submissions/final_submission_02.csv')


# FEATURES -----------------------------------------------------------------

# DL12 features
dl1 = fread(DL1_FILE); dl2 = fread(DL2_FILE)
dl12 = merge(dl1, dl2, by=c('patientid','nslice','x','y','diameter'))
dl12[,score:=(score.x+score.y)/2]
#dl12[,score:=score.x]
dl12 = dl12[score>0.7]
dl12_final = dl12[,.(max_diameter_patches = max(diameter, na.rm=T),
                     max_score_patches = max(score, na.rm=T)),
                  by=patientid]


# AUGMENTED features (previous with aggregation and augmentation)
df_nodule = fread(NODULES_EXTENDED_FILE)
df_nodule = plyr::ddply(
  df_nodule, "patientid", 
  function(df) {
    df_max = df[which.max(df$score), ]
    df_max$patientid = NULL
    df_max$nslicespread_sd = sd(df$nslicesSpread) 
    names(df_max) = paste0("maxscorenod_", names(df_max))

    df_maxd = df[which.max(df$diameter), ]
    df_maxd$patientid = NULL
    names(df_maxd) = paste0("maxdiamnod_", names(df_maxd))
    
    df_maxsS = df[which.max(df$nslicesSpread), ]
    df_maxsS$patientid = NULL
    names(df_maxsS) = paste0("maxnslicesSpread_", names(df_maxsS))

    data.table(df_max, df_maxd, df_maxsS, stringsAsFactors = FALSE)
  }, .progress = "text"
) %>% as.data.table
df_nodule = df_nodule[,.(patientid, maxscorenod_score, maxscorenod_PC1_lbp , maxscorenod_PC3_lbp, 
                         maxdiamnod_40_nodeverticalposition, maxdiamnod_PC2_hog,
                         maxnslicesSpread_nslicesSpread, maxnslicesSpread_score, maxnslicesSpread_09_inertia_tensor_eigvals1, maxnslicesSpread_40_nodeverticalposition)]


# DL3 features (just used for model 02)
dl3_df = fread(DL3_FILE)
dl3_df = dl3_df[,.(dl3_max=max(score)),by=patientid]


# model 01 TRAIN -------------------------------------------------------------------

# create data train df
data_train = fread(ANNOTATIONS_FILE)
setnames(data_train,'id','patientid')
data_train[,patientid:=paste0("dsb_",patientid,".npz")]
data_train = merge(data_train, dl12_final, by='patientid', all.x=T)
data_train = merge(data_train, df_nodule, by='patientid', all.x=T)
data_train[is.na(data_train)] = 0

vars_sel = c('max_diameter_patches', 'maxscorenod_score', 'maxscorenod_PC1_lbp' , 'maxscorenod_PC3_lbp', 'maxdiamnod_40_nodeverticalposition', 
             'maxdiamnod_PC2_hog', 'maxnslicesSpread_nslicesSpread', 'maxnslicesSpread_score', 'maxnslicesSpread_09_inertia_tensor_eigvals1',
             'maxnslicesSpread_40_nodeverticalposition')

aucs = c(); lls = c()
for(i in 1:100){
  k = 4; data_train$id = sample(1:k, nrow(data_train), replace = TRUE); list = 1:k
  for (i in 1:k){
    trainingset = subset(data_train, id %in% list[-i])
    testset = subset(data_train, id %in% c(i))

    # train
    mymodel = glm(cancer ~ ., family=binomial(link='logit'), data=trainingset[,c("cancer",vars_sel),with=F])

    # test
    pred = predict(mymodel, testset[,c("cancer",vars_sel),with=F], type="response")
    real = testset$cancer

    # store results
    aucs = c(aucs, auc(real,pred))
    lls = c(lls, MultiLogLoss(real,pred))
  }
}
summary(mymodel); summary(aucs); summary(lls)


# store final model
final_model_01 = glm(cancer ~ ., family=binomial(link='logit'), data=data_train[,c("cancer",vars_sel),with=F])
summary(final_model_01)
save(final_model_01, file=FINAL_MODEL_01)

#   (Intercept)                                 -80.767850  27.159102  -2.974  0.00294 ** 
#   max_diameter_patches                          0.038559   0.006678   5.774 7.76e-09 ***
#   maxscorenod_score                            75.216320  27.555970   2.730  0.00634 ** 
#   maxscorenod_PC1_lbp                           3.584367   0.611717   5.860 4.64e-09 ***
#   maxscorenod_PC3_lbp                          -4.442433   1.476519  -3.009  0.00262 ** 
#   maxdiamnod_40_nodeverticalposition           -0.751169   0.266787  -2.816  0.00487 ** 
#   maxdiamnod_PC2_hog                           -2.830997   1.180348  -2.398  0.01646 *  
#   maxnslicesSpread_nslicesSpread                0.161173   0.030162   5.344 9.11e-08 ***
#   maxnslicesSpread_score                        3.500830   1.459225   2.399  0.01644 *  
#   maxnslicesSpread_09_inertia_tensor_eigvals1   0.033030   0.010980   3.008  0.00263 ** 
#   maxnslicesSpread_40_nodeverticalposition     -0.917871   0.227924  -4.027 5.65e-05 ***



# model 01 EXECUTION ------------------------------------------------------

# construct table test
data_test = fread(SUBMISSIONS_FILE)
setnames(data_test,'id','patientid')
data_test[,patientid:=paste0("dsb_",patientid,".npz")]
data_test = merge(data_test, dl12_final, by='patientid', all.x=T)
data_test = merge(data_test, df_nodule, by='patientid', all.x=T)  
data_test[is.na(data_test)] = 0


load(FINAL_MODEL_01)
summary(final_model_01)
vars_sel01 = c('max_diameter_patches', 'maxscorenod_score', 'maxscorenod_PC1_lbp' , 'maxscorenod_PC3_lbp', 'maxdiamnod_40_nodeverticalposition', 
               'maxdiamnod_PC2_hog', 'maxnslicesSpread_nslicesSpread', 'maxnslicesSpread_score', 'maxnslicesSpread_09_inertia_tensor_eigvals1',
               'maxnslicesSpread_40_nodeverticalposition')
preds01 = predict(final_model_01, data_test[,vars_sel01,with=F], type="response")
submission01 = data.table(id=gsub(".npz|dsb_","",data_test$patientid), cancer=preds01)
cat("Mean cancer in submission01:", mean(submission01$cancer),"\n")
write.csv(submission01, "/home/shared/output/submissions/stage2_final_submission_mod01.csv", quote=F, row.names=F)
#write.csv(submission01, SUBMISSION_OUTPUT01, quote=F, row.names=F)


# check submission
dd = fread("/home/mingot/lung_cancer_ds_bowl/data/stage1_solution.csv")
dd = merge(submission01, dd, by="id")
MultiLogLoss(dd$cancer.y, dd$cancer.x)
  
# data_test[,preds:=preds01]
# data_test[,cancer:=NULL]
# dd = fread("/home/mingot/lung_cancer_ds_bowl/data/stage1_solution.csv")
# dd[,patientid:=paste0('dsb_',id,'.npz')]
# dd[,id:=NULL]
# data_test = merge(data_test, dd, by="patientid")
# 
# testdl11old = copy(data_test)
# testdl11new = copy(data_test)
# 
# xx = merge(testdl11old[,.(patientid, preds_old=preds, cancer)], testdl11new[,.(patientid, preds_new=preds)], by='patientid')
# xx[,dif:=abs(preds_old-preds_new)]
# xx[order(-dif)][0:20]
# pid = 'dsb_9cc74e673ec9807ee055973e1b185624.npz'
# testdl11old[patientid==pid]; testdl11new[patientid==pid]; 

# model 02 TRAIN ----------------------------------------------------------------

# validation patients
validation = fread(DL3_VALIDATION_FILE)

# create data train df
data_train = fread(ANNOTATIONS_FILE)
setnames(data_train,'id','patientid')
data_train[,patientid:=paste0("dsb_",patientid,".npz")]
data_train = merge(data_train, dl12_final, by='patientid', all.x=T)
data_train = merge(data_train, df_nodule, by='patientid', all.x=T)  # TODO: check if we are getting all the filenames
data_train = merge(data_train, dl3_df, by='patientid', all.x=T)
data_train[is.na(data_train)] = 0  # replace NA's with 0's
data_train = data_train[patientid%in%validation$patientid]

# to impose mean of 0.26, remove 18 1's
# n = (169 - 0.26*598)/(1  - 0.26)
# n = (112 - 0.2591267*400)/(1 - 0.2591267)
# remove = data_train[cancer==1][sample(.N,11)]$patientid
remove = data_train[cancer==1][sample(.N,16)]$patientid
data_train = data_train[!patientid%in%remove]
mean(data_train$cancer)


vars_sel = c('max_diameter_patches', 'maxscorenod_score', 'maxscorenod_PC1_lbp' , 'maxscorenod_PC3_lbp', 'maxdiamnod_40_nodeverticalposition', 
             'maxdiamnod_PC2_hog', 'maxnslicesSpread_nslicesSpread', 'maxnslicesSpread_score', 'maxnslicesSpread_09_inertia_tensor_eigvals1',
             'maxnslicesSpread_40_nodeverticalposition')  # MODEL 01

vars_sel = c('max_diameter_patches', 'maxscorenod_score', 'maxscorenod_PC1_lbp' , 
             'maxdiamnod_PC2_hog', 'maxnslicesSpread_score', 
             'maxnslicesSpread_40_nodeverticalposition','dl3_max')  


aucs = c(); lls = c()
for(i in 1:100){
  k = 3; data_train$id = sample(1:k, nrow(data_train), replace = TRUE); list = 1:k
  for (i in 1:k){
    trainingset = subset(data_train, id %in% list[-i])
    testset = subset(data_train, id %in% c(i))

    # train
    mymodel = glm(cancer ~ 1 + ., family=binomial(link='logit'), data=trainingset[,c("cancer",vars_sel),with=F])

    # test
    pred = predict(mymodel, testset[,c("cancer",vars_sel),with=F], type="response")
    real = testset$cancer

    # store results
    aucs = c(aucs, auc(real,pred))
    lls = c(lls, MultiLogLoss(real,pred))
  }
}
summary(mymodel); summary(aucs); summary(lls)

#   (Intercept)                              -124.93706   53.22096  -2.348  0.01890 *  
#   max_diameter_patches                        0.03657    0.01172   3.121  0.00180 ** 
#   maxscorenod_score                         116.83963   53.91903   2.167  0.03024 *  
#   maxscorenod_PC1_lbp                         2.94295    1.00645   2.924  0.00345 ** 
#   maxdiamnod_PC2_hog                         -4.60222    2.10257  -2.189  0.02861 *  
#   maxnslicesSpread_score                      5.62041    2.66333   2.110  0.03483 *  
#   maxnslicesSpread_40_nodeverticalposition   -1.53079    0.37837  -4.046 5.22e-05 ***
#   dl3_max                                     2.98118    0.62889   4.740 2.13e-06 ***

#   (Intercept)                              -183.22615   62.83232  -2.916 0.003544 ** 
#   max_diameter_patches                        0.03518    0.01169   3.009 0.002622 ** 
#   maxscorenod_score                         175.27009   63.53558   2.759 0.005805 ** 
#   maxscorenod_PC1_lbp                         3.02565    1.00901   2.999 0.002712 ** 
#   maxdiamnod_PC2_hog                         -5.64079    2.11242  -2.670 0.007578 ** 
#   maxnslicesSpread_score                      5.16263    2.66326   1.938 0.052566 .  
#   maxnslicesSpread_40_nodeverticalposition   -1.27554    0.37335  -3.416 0.000634 ***
#   dl3_max                                     3.28208    0.68344   4.802 1.57e-06 ***


# store final model
final_model_02 = glm(cancer ~ 1 + ., family=binomial(link='logit'), 
                     data=data_train[,c("cancer",vars_sel),with=F])
summary(final_model_02)
save(final_model_02, file=FINAL_MODEL_02)



# model 02 EXECUTION ------------------------------------------------------

data_test = fread(SUBMISSIONS_FILE)
setnames(data_test,'id','patientid')
data_test[,patientid:=paste0("dsb_",patientid,".npz")]
data_test = merge(data_test, dl12_final, by='patientid', all.x=T)
data_test = merge(data_test, df_nodule, by='patientid', all.x=T)
data_test = merge(data_test, dl3_df, by='patientid', all.x=T)
data_test[is.na(data_test)] = 0

load(FINAL_MODEL_02)
vars_sel02 = c('max_diameter_patches', 'maxscorenod_score', 'maxscorenod_PC1_lbp', 
               'maxdiamnod_PC2_hog', 'maxnslicesSpread_score',
               'maxnslicesSpread_40_nodeverticalposition','dl3_max')
preds02 = predict(final_model_02, data_test[,vars_sel02,with=F], type="response")
submission02 = data.table(id=gsub(".npz|dsb_","",data_test$patientid), cancer=preds02)
cat("Mean cancer in submission02:", mean(submission02$cancer),"\n")
# submission02[,cancer:=cancer-mean(cancer)+0.2626959]  # fix bias in the 600 patients data
write.csv(submission02, "/home/shared/output/submissions/stage2_final_submission_mod02.csv", quote=F, row.names=F)
#write.csv(submission02, SUBMISSION_OUTPUT02, quote=F, row.names=F)


# check submission
dd = fread("/home/mingot/lung_cancer_ds_bowl/data/stage1_solution.csv")
dd = merge(submission02, dd, by="id")
dd[,dif:=abs(cancer.x-cancer.y)]
dd[order(-dif)][1:20][,.(id)]

MultiLogLoss(dd$cancer.y, dd$cancer.x)


# Feature selection -------------------------------------------------------
library(FSelector)
library(ggplot2)
library(MASS)

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
var_importance = function(formula,data,num.vars=10){
  # Given data (frame/table) and a formula to get the target
  # it prints the top variables based on different criteria
  #
  # Args:
  #   formula: target ~ feature1 + feature2
  #   data: data (frame/table) with features and the target variable
  #
  # Returns:
  #   Prints the top 10 variables based on different criteria
  
  print('*************************************')
  print('LINEAR CORRELATION:')
  weights = linear.correlation(formula, data)
  weights$var = rownames(weights)
  print(weights[order(-weights$attr_importance)[1:num.vars],])
  
  print('*************************************')
  print('RANK CORRELATION:')
  weights = rank.correlation(formula, data)
  weights$var = rownames(weights)
  print(weights[order(-weights$attr_importance)[1:num.vars],])
  
  print('*************************************')
  print('INFORMATION GAIN:')
  weights = information.gain(formula, data)
  weights$var = rownames(weights)
  print(weights[order(-weights$attr_importance)[1:num.vars],])
  
  print('*************************************')
  print('GAIN RATIO:')
  weights = gain.ratio(formula, data)
  weights$var = rownames(weights)
  print(weights[order(-weights$attr_importance)[1:num.vars],])
  #   
  #   print('*************************************')
  #   print('RF ACCURACY:')
  #   weights = random.forest.importance(formula, data, importance.type=1)
  #   weights$var = rownames(weights)
  #   print(weights[order(-weights$attr_importance)[1:10],])
  #   
  #   print('*************************************')
  #   print('RF NODE IMPURITY:')
  #   weights = random.forest.importance(formula, data, importance.type=2)
  #   weights$var = rownames(weights)
  #   print(weights[order(-weights$attr_importance)[1:10],])
  
}

arcvi_descriptivos(data_train$dl3_max, data_train$cancer, equidistributed=F)
var_importance(cancer~., data=data_train[,!names(data_train)%in%c('id','patientid'),with=F])

fit = glm(cancer ~ 1 + ., family=binomial(link='logit'), data=data_train[,c("cancer",vars_sel),with=F])
step = stepAIC(fit, direction="both")

