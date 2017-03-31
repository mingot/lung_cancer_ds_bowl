library(data.table)
library(pls)
library(spls)
# library(mixOmics)
library(plyr)
library(dplyr)
library(caret)
# library(mlr)
library(Metrics)

set.seed(1)

# submission df
# patient format is bad - will be improved
df_submit <- read.csv("../data/stage1_sample_submission.csv", stringsAsFactors = FALSE)
colnames(df_submit) <- c("patientid", "cancer")

# labels
df_cancer <- read.csv("../data/stage1_labels.csv", stringsAsFactors = FALSE)
colnames(df_cancer) <- c("patientid", "cancer")


# df_nodule <- fread("../data/output_dl_example_augmented.csv")
df_nodule <- fread("../data/nodules_v06_score07_augmented.csv")
df_nodule[, `Unnamed: 0` := NULL]
df_nodule$patientid <- gsub("(.+)(dsb_|luna_)([[:alnum:]]+)(\\.npz)", "\\3", df_nodule$patientid)

# aggregate patients
df_nodule %<>% 
  group_by(patientid) %>% 
  summarise_each(funs(length, min, median, mean, max, sd)) %>% 
  as.data.table

# any missing patient?
sum(df_submit$patientid %in% df_nodule$patientid)
dim(df_submit)

# background proportion for missing patient imputation
NA_prop <- mean(df_cancer$cancer)
df_submit$cancer <- NA_prop

# remove features that will have no variance
col_length <- grep(".+_length$", colnames(df_nodule), value = TRUE)
df_nodule$n_patches <- df_nodule[[col_length[1]]]
df_nodule[, (col_length) := NULL]
df_nodule[, c("10_lungmask_min", "10_lungmask_median", "10_lungmask_max", "10_lungmask_sd") := NULL]

# some sd can be na: collapse them to 0
sum(is.na(df_nodule))
df_nodule[is.na(df_nodule)] <- 0

# save features
write.csv(df_nodule, file = "submissions/sp_01_features.csv", row.names = F, quote = F)

# inner join, add cancer column
df_model <- df_nodule[df_cancer, on = "patientid", nomatch = 0]
#df_model$cancer <- as.factor(ifelse(df_model$cancer, "yes", "no"))


# Split train-test
set.seed(2)
inTraining <- caret::createDataPartition(df_model$cancer, p = .85, list = FALSE)
# lapply(row_train, length)
training <- df_model[ inTraining, !("patientid"), with = FALSE]
testing  <- df_model[-inTraining, !("patientid"), with = FALSE]

# To assess performance, cv in train
set.seed(3)
mod_splscv <- spls::cv.splsda(
  x = as.matrix(training[, !"cancer"]), 
  y = training$cancer, 
  K = 1:8, 
  eta = seq(from = .6, to = .95, by = .05), 
  classifier = "logistic", 
  n.core = 4
)
mod_spls <- spls::splsda(
  x = as.matrix(training[, !"cancer"]), 
  y = training$cancer, 
  K = mod_splscv$K.opt, 
  eta = mod_splscv$eta.opt, 
  classifier = "logistic")
mod_spls

# predict test set
ypred <- spls::predict.splsda(
  mod_spls, 
  testing[, !"cancer"], 
  fit.type = "response"
)

# quality metrics
Metrics::logLoss(testing$cancer, ypred)
Metrics::auc(testing$cancer, ypred)

# Final model
# Train with all the samples (cv for param optim)
set.seed(4)
mod_splscv_final <- spls::cv.splsda(
  x = as.matrix(df_model[, !c("cancer", "patientid")]), 
  y = df_model$cancer, 
  K = 1:8, 
  eta = seq(from = .6, to = .95, by = .05), 
  classifier = "logistic", 
  n.core = 4
)
mod_spls_final <- spls::splsda(
  x = as.matrix(df_model[, !c("cancer", "patientid")]), 
  y = df_model$cancer, 
  K = mod_splscv_final$K.opt, 
  eta = mod_splscv_final$eta.opt, 
  classifier = "logistic")
mod_spls_final

# save results
df_eval <- df_nodule[patientid %in% df_submit$patientid]
pred_out <- spls::predict.splsda(
  mod_spls_final, 
  newx = df_eval[, !"patientid"], 
  fit.type = "response"
)
names(pred_out) <- df_eval$patientid

pat_na <- setdiff(df_submit$patientid, df_eval$patientid)
v_na <- setNames(rep(NA_prop, length(pat_na)), pat_na)

v_out <- c(pred_out, v_na)
df_out <- data.frame(
  id = names(v_out), 
  cancer = v_out, 
  stringsAsFactors = FALSE
)
write.csv(df_out, file = "submissions/sp_01_splsda.csv", row.names = FALSE, quote = FALSE)


# SANITY CHECKS and stuff
# 
# x <- read.csv("~/Downloads/05_submission.csv", stringsAsFactors = FALSE)
# all(df_out$id %in% x$id)
# all(dim(df_out) == dim(x))
# 
# 
# 
# # See used features when pls is forced to sparsify 
# set.seed(3)
# mod_splscv <- spls::cv.splsda(
#   x = as.matrix(training[, !"cancer"]), 
#   y = training$cancer, 
#   K = 1:8, 
#   eta = .9, 
#   classifier = "logistic", 
#   n.core = 4
# )
# mod_spls <- spls::splsda(
#   x = as.matrix(training[, !"cancer"]), 
#   y = training$cancer, 
#   K = mod_splscv$K.opt, 
#   eta = mod_splscv$eta.opt, 
#   classifier = "logistic")
# mod_spls
# 
# ypred <- spls::predict.splsda(
#   mod_spls, 
#   testing[, !"cancer"], 
#   fit.type = "response"
# )
# 
# Metrics::logLoss(testing$cancer, ypred)
# Metrics::auc(testing$cancer, ypred)




# mixomics
# 
# PETA! 
# 
# library(mixOmics)
# 
# mo_splsda <- mixOmics::tune.splsda(
#   X = as.matrix(training[, !"cancer"]),
#   Y = ifelse(training$cancer == "yes", 1, 0),
#   ncomp = 3,
#   test.keepX = c(5, 10),
#   folds = 3,
#   progressBar = TRUE)

