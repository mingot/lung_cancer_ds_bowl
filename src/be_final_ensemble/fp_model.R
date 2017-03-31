# Reads a .csv containing nodules (and optionally extra features for those nodules) and creates a model 
# for reducing the number of FPs.
luna_data = fread(paste0(path_repo,"/data/final_model/output_model_teixi_luna.csv"))
luna_data = luna_data[(!x %in% extreme_values) & (!y %in% extreme_values)] 

#hog = data.table(read.csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/hog_v1.csv', header=F))
hog = fread(paste0(path_repo,"/data/final_model/hog_v2_total.csv"), header=F)
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
vars_model <- c("nslice","x","y","diameter","max_intensity","min_intensity","mean_intensity","target")
luna_data <- luna_data[,.SD,.SDcols = vars_model]
luna_data[,target := as.factor(target)]

train_task <- makeClassifTask(data = data.frame(luna_data),target = "target")
lrn = generateModel("classif.logreg")$lrn
k = 5 #Folds
rdesc = makeResampleDesc("CV", iters = k, stratify = TRUE)

# K-fold and training
parallelStartSocket(5)
tr_cv = resample(lrn, train_task, rdesc, models = TRUE,measures = list(auc,logloss,fpr,fnr))
knitr::knit_print(tr_cv$measures.test)
fp_model = train(lrn,train_task)
parallelStop()

