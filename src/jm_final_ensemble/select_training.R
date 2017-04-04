select_training <- function(dataset,score) {
  dataset[,cancer := NULL]
  score[,cancer := NULL]
  dataset[,target := 1]
  score[,target := 0]
  
  total <- rbind(dataset,score)
  total[,target := as.factor(target)]
  train_task = makeClassifTask(data = data.frame(total), target = "target")
  # fv <- generateFilterValuesData(data_task, method = "chi.squared")
  # features_null <- data.table(fv$data)[chi.squared == 0,name]
  
  #train_task <- makeClassifTask(data = data.frame(total), target = "target")
  
  model_family <- "classif.logreg" 
  model_list <- generateModel(model_family)
  lrn <- model_list$lrn
  ps <- model_list$ps
  
  
  n_cv <- 2L
  rdesc = makeResampleDesc("CV", iters = n_cv, stratify = TRUE)
  #tr = train(lrn, train_task)
  parallelStartSocket(2)
  tr = resample(lrn, train_task, rdesc, models = TRUE)
  parallelStop()
  
  predictions <- predictCv(tr,train_task)
  print(my.AUC(as.numeric(as.character(total[,target])),predictions))
  total$pred <- predictions
  total[,target := as.numeric(as.character(target))]
  return(total)
}


# indexes <- select_training(copy(data_train),copy(scoring))
# indexes_train <- indexes[,.I[target == 1 & pred < 0.87]]
# data_test <- data_train[!indexes_train]
# data_train <- data_train[indexes_train]