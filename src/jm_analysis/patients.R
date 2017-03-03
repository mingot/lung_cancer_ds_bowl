library(data.table)

annotations = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/stage1_labels.csv'))
data = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/dl_unet_v01.csv'))
data[,patientid:=gsub("dsb_","",patientid)]
data = merge(data, annotations, by.x="patientid", by.y="id", all.x=T)
data = data[!is.na(cancer)]  # remove test (for which we do not have cancer label)

luna_nodules = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/annotations.csv'))
summary(luna_nodules$diameter_mm)
quantile(luna_nodules$diameter_mm,0.9)

# FILTER 1: extreme windows (1.5M -> 1.1M)
data = data[x!=0 & x!=511 & y!=0 & y!=511]  

# FILTER 2: small and large nodules (1.1M -> 0.78M)
# From cancer paper:
# Benign nodule size (mm): mean: 4.1+-3.1, median: 3.4, quartiles: 2.7-5.0
# Malign nodule size (mm): mean: 15.7+-12.2, median: 13, quartiles: 8-19
# each pixel is 7mm. We accept diameter in [2.5mm, 20mm] -> [3.5, 28] pixel units of diameter
summary(data$diameter)
data = data[diameter %between% c(3.5, 28)]


# just high intensity nodules
data = data[mean_intensity>0.8]

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

# nodules distribution
library(MASS)
pid = sample(unique(data$patientid),1)
dens = kde2d(data[patientid==pid]$x, data[patientid==pid]$y)
filled.contour(dens)
xx[patientid==pid]

# nodules per person
xx = data[,.(total_nodules=.N, nodules_per_slice=.N/uniqueN(nslice),
             num_slices=uniqueN(nslice), 
             max_intensity=max(max_intensity), max_mean_intensity=max(mean_intensity)),
          by=.(patientid,cancer)]

arcvi_descriptivos(xx$N, xx$cancer, equidistributed=F)
arcvi_descriptivos(xx[max_mean_intensity>0.853]$max_mean_intensity, xx[max_mean_intensity>0.853]$cancer, equidistributed=F)

xx[,high_intensity:=ifelse(max_mean_intensity>0.9,1,0)]
xx[,lot_nodules:=ifelse(total_nodules>150,1,0)]
xx[,few_nodules:=ifelse(total_nodules<50,1,0)]

auc(xx$cancer, xx$lot_nodules)  # 0.51
auc(xx$cancer, xx$few_nodules)  # 0.49
auc(xx$cancer, xx$high_intensity)  # 0.49

model = glm(cancer ~. - patientid, family=binomial(link='logit'), data=xx)
summary(model)



k = 4 #Folds
xx$id = sample(1:k, nrow(xx), replace = TRUE)
list = 1:k

for (i in 1:k){
  trainingset = subset(xx, id %in% list[-i])
  testset = subset(xx, id %in% c(i))
  
  # train
  #mymodel = randomForest(trainingset$target ~ ., data = trainingset, ntree = 100)
  mymodel = glm(cancer ~. - patientid, family=binomial(link='logit'), data=xx)
  
  # test
  pred = predict(mymodel, testset)
  real = testset$cancer

cat(auc(real,pred),"\n")
}

