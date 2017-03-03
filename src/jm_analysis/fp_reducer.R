library(data.table)


data = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/output_model_teixi_luna.csv'))
data = data[x!=0 & x!=511 & y!=0 & y!=511] 

hog = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/hog_v1.csv', header=F))
setnames(hog, c('V1','V2','V3','V4','V5'), c('score','filename','nslice','x','y'))


data = merge(data, hog, by=c("filename","nslice","x","y"), all.x=T)

# TEMPORAL!!
data = data[!is.na(score)]
data[,target:=as.numeric(score>0.3)]

mymodel = glm(target ~. - filename - score, family=binomial(link='logit'), data=data)

k = 4 #Folds
xx$id = sample(1:k, nrow(xx), replace = TRUE)
list = 1:k

for (i in 1:k){
  trainingset = subset(xx, id %in% list[-i])
  testset = subset(xx, id %in% c(i))
  
  # train
  #mymodel = randomForest(trainingset$target ~ ., data = trainingset, ntree = 100)
  mymodel = glm(target ~. - filename - score, family=binomial(link='logit'), data=xx)
  
  # test
  pred = predict(mymodel, testset)
  real = testset$cancer
  
  cat(auc(real,pred),"\n")
}


