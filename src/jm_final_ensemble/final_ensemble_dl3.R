library(data.table)
library(pROC)
library(gbm)
#library(caret)


path_repo <<- "~/lung_cancer_ds_bowl/" #"D:/lung_cancer_ds_bowl/"
path_dsb <<- "/home/shared/output/" #"D:/dsb/"

validation = fread(paste0(path_repo, 'data/stage1_validation.csv'))
validation[,patientid:=gsub(".npz|dsb_","",patientid)]

# load DL3 features
dl3_df = fread(paste0(path_dsb,"resnet/nodules_patches_dl3_v02.csv"))
dl3_df = dl3_df[,.(dl3_max=max(score), 
                   dl3_has=as.numeric(sum(score>0.7)>0),
                   dl3_nohas=as.numeric(sum(score<.3)==.N)
                   ),by=patientid]
dl3_df[,patientid:=gsub(".npz|dsb_","",patientid)]

# Contruct partial training set (over 993 patients)
xx = fread(paste0(path_dsb,'dl3_train/partial_submission_16.csv'))
data_train = dt
data_train[,patientid:=gsub(".npz|dsb_","",patientid)]
data_train = merge(data_train, dl3_df, by='patientid', all.x=T)
data_train[is.na(dl3_nohas), dl3_nohas:=1]
data_train[is.na(data_train)] = 0  # replace NA's with 0's
data_train = data_train[patientid%in%xx$patientid]
View(data_train)

# Cross validation --------------------------------------------------------

MultiLogLoss = function(act, pred){
  eps = 1e-15
  pred = pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

vars_sel = c('max_diameter_patches', 'max_nsliceSpread', 'max_score_patches', 'PC1_lbp', 'man_nslicespread_sd', 'dl3_has')#, 'dl3_has', 'dl3_nohas')

aucs = c()
lls = c()
for (j in 1:50){
k = 3 #Folds
data_train$id = sample(1:k, nrow(data_train), replace = TRUE)
list = 1:k
  for (i in 1:k){
    trainingset = subset(data_train, id %in% list[-i])
    testset = subset(data_train, id %in% c(i))
    
    # train
    mymodel = glm(cancer ~ ., family=binomial(link='logit'),
                  data=trainingset[,c("cancer",vars_sel),with=F])
  
    # test
    pred = predict(mymodel, testset[,c("cancer",vars_sel),with=F], type="response")#, n.trees=100)
    real = testset$cancer
    
    # store results
    aucs = c(aucs, auc(real,pred))
    lls = c(lls, MultiLogLoss(real,pred))
  }}
summary(mymodel)
summary(aucs)
summary(lls)


trainingset = data_train
mymodel = glm(cancer ~ . -dl3_max, family=binomial(link='logit'), 
              data=trainingset[,c("cancer",vars_sel),with=F])
summary(mymodel)


# final model -------------------------------------------------------------

# dl3_final_df = fread(paste0(path_dsb,'resnet/nodules_patches_dl3_v03.csv'))
# dl3_final_df = dl3_final_df[,.(dl3_max=max(score), dl3_num_maligns=sum(score>0.2)),by=patientid]
# dl3_final_df[,patientid:=gsub(".npz|dsb_","",patientid)]
# 
# hist(data_submission$dl3_max.y)
# hist(data_submission$dl3_max.x)

data_submission = fread(paste0(path_dsb,'dl3_train/total_submission_16.csv'))
data_submission = merge(data_submission, dl3_df, by='patientid', all.x=T)
# data_submission = merge(data_submission, dl3_final_df, by='patientid', all.x=T)
data_submission[is.na(dl3_max), dl3_max:=0]
data_submission[is.na(dl3_has), dl3_has:=0]

preds = predict(mymodel, data_submission[,!names(testset)%in%'patientid',with=F], type="response")
submission_df = data.table(id=data_submission$patientid, cancer=preds)
mean(submission_df$cancer)
write.csv(submission, paste0(path_repo,"data/submissions/21_submission.csv"), quote=F, row.names=F)



# visualization -------------------------------------------------------------

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



arcvi_descriptivos(data_train$dl3_max, data_train$cancer)
arcvi_descriptivos(data_train$dl3_has, data_train$cancer)
arcvi_descriptivos(data_train$dl3_nohas, data_train$cancer)
