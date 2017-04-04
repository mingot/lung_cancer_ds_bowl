library(data.table)
# library(pls)
library(spls)
# # library(mixOmics)
library(plyr)
library(dplyr)
library(caret)
# library(magrittr)
# # library(mlr)
# library(Metrics)
# 
# set.seed(1)
# 
# # submission df
# # patient format is bad - will be improved

# 
# labels
source("aux_autopls.R")

strip_dsb <- function(x) {
  gsub("(dsb_|luna_)([[:alnum:]]+)(\\.npz)", "\\2", x)
}

source("aux_benchmark.R")

# 
# 
df_magic <- fread("../../data/magic.csv")
df_magic[, c("cancer", "dataset") := NULL]

df_nodule <- fread("../../data/nodules_patches_dl1_v11_score07_noduleaggr_augmented.csv")
df_nodule[, c("Unnamed: 0", "Unnamed: 0.1") := NULL]
df_nodule[, patientid := strip_dsb(patientid)]

# df_nodule <- fread("../../data/nodules_patches_dl1_v11_score07_augmented.csv")
# df_nodule[, patientid := strip_dsb(patientid)]

plot(df_nodule$diameter, df_nodule$score, pch = ".")
 # filter some of the nodules
sum(df_nodule$diameter < 4.5)/nrow(df_nodule)
df_nodule <- df_nodule[diameter > 5]
hist(df_nodule$`03_area`)
hist(df_nodule$`10_lungmask`)
sum(df_nodule$`03_area`< 50) 
sum(.3*pi*12.5**2 < df_nodule$`03_area`)
df_nodule <- df_nodule[.3*pi*12.5**2 < df_nodule$`03_area`]
# transform mask to binary
df_nodule[, `10_lungmask` := (`10_lungmask` == 1)*1] 
hist(df_nodule$`40_nodeverticalposition`)
hist(df_nodule$`04_perimeter`)
df_nodule <- df_nodule[`04_perimeter` > 30]

# df_nodule %<>%
#   group_by(patientid) %>%
#   summarise_each(funs(length, min, mean, max, sd)) %>%
#   as.data.table
df_nodule <- plyr::ddply(
  df_nodule, 
  "patientid", 
  function(df)  {
    # browser()
    df_top <- dplyr::arrange(df, desc(score)) %>% head(1) %>% as.data.table
    df_out <- df_top[, !"patientid"] #%>% summarise_each(funs(min, mean, max, sd))
    df_out$man_upperLung <- sum(df$"40_nodeverticalposition" < .3 & df$score > 0.95)
    df_out$man_propWall <- mean(df$"10_lungmask")
    df_out$man_mean_area <- mean(df$"03_area")
    df_out$man_bigdiam <- sum(df$diameter > 20)
    df_out$man_lowerLung <- sum(df$"40_nodeverticalposition" > .7 & df$score > 0.95)
    df_out$man_nslicespread_sd <- sd(df$nslicesSpread)
    
    df_out
  }, .progress = "text"
) %>% as.data.table


# 

# 
# # background proportion for missing patient imputation
# NA_prop <- mean(df_cancer$cancer)
# df_submit$cancer <- NA_prop
# 
# # remove features that will have no variance
# col_length <- grep(".+_length$", colnames(df_nodule), value = TRUE)
# df_nodule$n_patches <- df_nodule[[col_length[1]]]
# df_nodule[, (col_length) := NULL]
# df_nodule[, c("10_lungmask_min", "10_lungmask_median", "10_lungmask_max", "10_lungmask_sd") := NULL]
# # 
# # # some sd can be na: collapse them to 0
# sum(is.na(df_nodule))
# summary(df_nodule)
# df_nodule[is.na(df_nodule)] <- 0
# 
# df_nodule[, score_max := (score_max < .999)*1]




# df_emphy <- fread("../../data/var_emphysema_v05.csv")
# colnames(df_emphy) <- c("patientid", paste0("emphy", 1:3))
# df_intercost <- fread("../../data/stage1_extra_features_intercostal.csv")
# df_intercost$patientid <- strip_dsb(df_intercost$patient_id)
# df_intercost$patient_id <- NULL
# 
df_cancer <- read.csv("../../data/stage1_labels.csv", stringsAsFactors = FALSE)
colnames(df_cancer) <- c("patientid", "cancer")

df_submit <- read.csv("../../data/stage1_sample_submission.csv", stringsAsFactors = FALSE)
colnames(df_submit) <- c("patientid", "cancer")
# 
# # intercostal
# dim(df_intercost)
# 
# # emphy
# df_all <- df_emphy[df_cancer, on = "patientid", nomatch = 0]
# dim(df_all)
# apply(df_all[, -c("patientid", "cancer")], 2, function(col) Metrics::auc(df_all$cancer, col))
# 
# # intercostal
# df_all <- df_intercost[df_cancer, on = "patientid", nomatch = 0]
# dim(df_all)
# apply(df_all[, -c("patientid", "cancer")], 2, function(col) Metrics::auc(df_all$cancer, col))
# 
# benchmark_feat(df_emphy, df_cancer)
# benchmark_feat(df_intercost, df_cancer)
# benchmark_feat(df_magic, df_cancer)
# 
# df_cols <- benchmark_feat(df_nodule, df_cancer)
# df_cols <- as.character(subset(df_cols, abs(auc - 0.5) > .05)$feature)

df_magic2 <- df_magic[df_nodule[, .(patientid, PC1_lbp, man_upperLung, man_lowerLung, man_nslicespread_sd)], on = "patientid", nomatch = 0]

# df_magic2 <- df_magic[df_nodule, on = "patientid", nomatch = 0]

set.seed(1)
out <- wrapper_submit_pls(
  # df_features = df_nodule[, c("patientid", df_cols), with=FALSE],
  # df_features = df_nodule,
  df_features = df_magic2, 
  df_cancer = df_cancer, 
  df_submit = df_submit, 
  out_submit = "submissions/sp_07_testing",
  # out_perf = "_perf", 
  cv_out_k = 4,
  cv_in_k = 4, 
  cv_out_times = 5, 
  K = 1:2, 
  eta = c(0,.2,.4)
  # eta = c(.8)
)
# out$perf$auc
summary(plyr::ddply(out$perf, "fold", function(x) x$logLoss[1])$V1)
summary(plyr::ddply(out$perf, "fold", function(x) x$auc[1])$V1)
out$model

