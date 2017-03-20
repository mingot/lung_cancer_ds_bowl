library(data.table)
library(pROC)


## nueva
luna_data = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/noduls_patches_v05_backup3.csv_output'))
luna_data[,target:=as.numeric(intersection_area>0.1)]  # dangerous!

## reduce 0's to make it balanced
cands = which(luna_data$target==0)
data0 = luna_data[sample(cands,0.001*length(cands))]
luna_data = rbind(luna_data[target==1], data0)
mean(luna_data$target)


## interesting variables about the position!!
# luna_data[,var_y:=as.numeric(y>145 & y<195)]
# luna_data[,var_x:=as.numeric(x>345)]
# arcvi_descriptivos(luna_data$y, luna_data$target)
# arcvi_descriptivos(luna_data$x, luna_data$target)


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
  fp_model = glm(target ~ 1 + var_y + var_x + diameter + score,
                 family=binomial(link='logit'), data=trainingset)
  
  # test
  pred = predict(fp_model, testset, type="response")
  real = testset$target
  
  cat(auc(real,pred),"\n")
  aucs = c(aucs, auc(real,pred))
}
mean(aucs)
sd(aucs)

