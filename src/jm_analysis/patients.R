library(data.table)

data = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/dl_unet_v01_mingot_pc.csv'))
data[,patientid:=gsub("dsb_","",patientid)]
# hog = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/hog_v1.csv', header=F))
# setnames(hog, c('V1','V2','V3','V4','V5'), c('score','patientid','nslice','x','y'))
# hog[,patientid:=gsub("luna_","",patientid)]
# hog[,patientid:=gsub(".npz","",patientid)]

# luna_nodules = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/annotations.csv'))
# summary(luna_nodules$diameter_mm)
# quantile(luna_nodules$diameter_mm,0.9)

# FILTER 1: extreme windows (1.5M -> 1.1M)
data = data[x!=0 & x!=511 & y!=0 & y!=511]  

# # FILTER 2: small and large nodules (1.1M -> 0.78M)
# # From cancer paper:
# # Benign nodule size (mm): mean: 4.1+-3.1, median: 3.4, quartiles: 2.7-5.0
# # Malign nodule size (mm): mean: 15.7+-12.2, median: 13, quartiles: 8-19
# # each pixel is 7mm. We accept diameter in [2.5mm, 20mm] -> [3.5, 28] pixel units of diameter
# summary(data$diameter)
# data = data[diameter %between% c(3.5, 28)]


# # FILTER 3: just high intensity nodules
# data = data[mean_intensity>0.8]

# FILTER 4: 
data[,nodule_pred:=predict(fp_model, data, type="response")]
data = data[nodule_pred>quantile(nodule_pred,0.9)]

## Add annotations
annotations = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/stage1_labels.csv'))
setnames(annotations, "id", "patientid")
data = merge(annotations, data, by="patientid", all.x=T)
data = data[!is.na(cancer)]  # remove test (for which we do not have cancer label)

mean(data[!is.na(nslice)]$cancer)
mean(data[is.na(nslice)]$cancer)


# data exploration --------------------------------------------------------
library(ggplot2)
library(pROC)

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

# nodules per person
final_df = data[,.(total_nodules=sum(!is.na(nslice)), 
                   num_slices=uniqueN(nslice[!is.na(nslice)]), 
                   nodules_per_slice=sum(!is.na(nslice))/uniqueN(nslice),
                   max_intensity=max(max_intensity, na.rm=T), 
                   max_mean_intensity=max(mean_intensity, na.rm=T)
                   #max_score=max(nodule_pred, na.rm=T),
                   #mean_score=mean(nodule_pred, na.rm=T)
                   ),
          by=.(patientid,cancer)]

final_df[!is.finite(max_intensity), max_intensity:=0]
final_df[!is.finite(max_mean_intensity), max_mean_intensity:=0]
final_df[!is.finite(max_score), max_score:=0]
final_df[!is.finite(mean_score), mean_score:=0]

arcvi_descriptivos(final_df$N, final_df$cancer, equidistributed=F)
arcvi_descriptivos(final_df[max_mean_intensity>0.853]$max_mean_intensity, 
                   final_df[max_mean_intensity>0.853]$cancer, equidistributed=F)

final_df[,high_intensity:=ifelse(max_mean_intensity>0.9,1,0)]
final_df[,lot_nodules:=ifelse(total_nodules>150,1,0)]
final_df[,few_nodules:=ifelse(total_nodules<50,1,0)]

auc(final_df$cancer, final_df$lot_nodules)  # 0.51
auc(final_df$cancer, final_df$few_nodules)  # 0.49
auc(final_df$cancer, final_df$high_intensity)  # 0.49

model = glm(cancer ~. - patientid, family=binomial(link='logit'), data=final_df)
summary(model)

final_df[,patientid:=as.character(patientid)]

k = 4 #Folds
final_df$id = sample(1:k, nrow(final_df), replace = TRUE)
list = 1:k


MultiLogLoss <- function(act, pred){
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}


for (i in 1:k){
  trainingset = subset(final_df, id %in% list[-i])
  testset = subset(final_df, id %in% c(i))
  
  # train
  #mymodel = randomForest(trainingset$target ~ ., data = trainingset, ntree = 100)
  mymodel = glm(cancer ~ . -id, family=binomial(link='logit'), 
                data=trainingset[,!names(testset)%in%'patientid',with=F])
  
  # test
  pred = predict(mymodel, testset[,!names(testset)%in%'patientid',with=F])
  real = testset$cancer

cat(auc(real,pred),"\n")
cat(MultiLogLoss(real,pred), "\n")
}

