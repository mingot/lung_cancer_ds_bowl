library(data.table)
library(pROC)

luna_data = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/output_model_teixi_luna.csv'))
luna_data = luna_data[x!=0 & x!=511 & y!=0 & y!=511] 

#hog = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/hog_v1.csv', header=F))
hog = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/hog_v2_total.csv', header=F))
setnames(hog, c('V1','V2','V3','V4','V5'), c('score','filename','nslice','x','y'))


luna_data = merge(luna_data, hog, by=c("filename","nslice","x","y"), all.x=T)

# TEMPORAL!!
luna_data = luna_data[!is.na(score)]
luna_data[,target:=as.numeric(score>0.1)]  # dangerous!

## reduce 0's to make it balanced
cands = which(luna_data$target==0)
data0 = luna_data[sample(cands,0.1*length(cands))]
luna_data = rbind(luna_data[target==1], data0)
mean(luna_data$target)


k = 3 #Folds
luna_data$id = sample(1:k, nrow(luna_data), replace = TRUE)
list = 1:k
aucs = c()

for (i in 1:k){
  trainingset = subset(luna_data, id %in% list[-i])
  testset = subset(luna_data, id %in% c(i))
  
  # train
  # fp_model = randomForest(trainingset$target ~ ., data = trainingset, ntree = 100)
  # fp_model = glm(target ~. - filename - score, family=binomial(link='logit'), data=data)
  fp_model = glm(target ~ 1 + nslice + x + y + diameter + max_intensity + min_intensity + mean_intensity,
                family=binomial(link='logit'), data=trainingset)

  # test
  pred = predict(fp_model, testset, type="response")
  real = testset$target
  
  cat(auc(real,pred),"\n")
  aucs = c(aucs, auc(real,pred))
}
mean(aucs)
sd(aucs)


