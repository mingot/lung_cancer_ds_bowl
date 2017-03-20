
library(data.table)
library(mlr)
library(parallelMap)
library(ggplot2)

dataset_slices <- fread("./output.csv")
dataset_slices <- dataset_slices[grep("dsb_",id)]
dataset_slices[,patient_id := gsub(".npz|dsb_","",patient_id)]
setnames(dataset_slices,"patient_id","patientid")

patients <- fread("stage1_labels.csv")
setnames(patients, "id", "patientid")
patients[,cancer := as.factor(cancer)]

dataset_final <- merge(patients,dataset_slices, all.y = T, by = "patientid")
dataset_final$patient_max[dataset_final$cancer==1]
dataset_final$patient_max[dataset_final$cancer==0]
hist(dataset_final$patient_max[dataset_final$cancer==1])
hist(dataset_final$patient_max[dataset_final$cancer==0])

summary(dataset_final$patient_max[dataset_final$cancer==1])
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.1833  0.2837  0.3324  0.3312  0.3713  0.4629     198 
summary(dataset_final$patient_max[dataset_final$cancer==0])
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.1261  0.2781  0.3242  0.3245  0.3686  0.4818     198 
