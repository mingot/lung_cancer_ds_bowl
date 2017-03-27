
generate_patient_dt <- function(path_repo,path_output = NULL) {
  annotations = fread(paste0(path_repo,"data/stage1_labels.csv"))
  submission = fread(paste0(path_repo,"/data/stage1_sample_submission.csv"))
  submission[,cancer := 0]
  submission[,dataset := "scoring"]
  annotations[,dataset := "training"]
  
  #Saving which patients are belong to training
  
  # Binding train and scoring
  patients <- rbind(annotations,submission)
  setnames(patients, "id", "patientid")
  patients[,cancer := as.factor(cancer)]
  
  
  ## UNET Data ---------------------------------------------------------------------------------------
  
  vars_nodules <- fread(paste0(path_repo,"data/final_model/dl_unet_v01_mingot_pc.csv"))
  vars_nodules[,patientid:=gsub("dsb_","",patientid)]
  vars_nodules = vars_nodules[(!x %in% extreme_values) & (!y %in% extreme_values)]  
  
  ### Filter by fp model
  vars_nodules[,nodule_pred:=predictCv(fp_model, vars_nodules)]
  vars_nodules = vars_nodules[nodule_pred>quantile(nodule_pred,0.9)]
  # Merging with patients to get cancer info for easier debug
  vars_nodules <- merge(vars_nodules,patients,all.x=T,by = "patientid")
  
  
  ## RESNET Data -------------------------------------------------------------------------------------
  #vars_nodules_patches <- fread(paste0("D:/dsb/nodules_patches_v05_augmented.csv"))
  vars_nodules_patches <- fread(paste0("D:/dsb/noduls_patches_v06_rectif.csv")) ## PATH
  vars_nodules_patches <- vars_nodules_patches[grep("dsb_",patientid)][!is.na(x)]
  vars_nodules_patches[,patientid:=gsub(".npz|dsb_","",patientid)]
  ### Filter by score
  vars_nodules_patches = vars_nodules_patches[score>0.9]
  names_change <- c("x","y","diameter","score")
  setnames(vars_nodules_patches,names_change,paste0(names_change,"_patches"))
  vars_nodules_patches <- merge(vars_nodules_patches,patients,all.x=T,by = "patientid")
  
  ## Merging al nodules variables
  vars_nodules <- rbind(vars_nodules,vars_nodules_patches,fill = TRUE)
  vars_nodules[,cancer := NULL]
  ## Aggregating to patient level
  dataset_nodules <- aggregate_patient(vars_nodules)
  
  ## SLICES OUTPUT Data ------------------------------------------------------------------------------
  
  dataset_slices <- fread(paste0(path_repo,"/src/dl_model_slices/output.csv"))
  dataset_slices <- dataset_slices[grep("dsb_",id)]
  setnames(dataset_slices,names(dataset_slices),paste0("patient_",names(dataset_slices)))
  dataset_slices[,patient_id := gsub(".npz|dsb_","",patient_id)]
  setnames(dataset_slices,"patient_id","patientid")
  
  
  ## Joining all the patient variables
  dataset_final <- merge(patients,dataset_nodules,all.x = T, by = "patientid")
  dataset_final <- merge(dataset_final,dataset_slices,all.x = T, by = "patientid")
  dataset_final <- na_to_zeros(dataset_final,names(dataset_final))
  if(is.null(path_output)) {
    return(dataset_final)
  } else {
    write.csv(dataset_final,paste0(path_output,"dataset_final.csv"),row.names=F)
    return(invisible())
  }

}

## EXTRA FUNCTIONS----------------------------------------------------------------------------------
aggregate_patient <- function(dt) {
  dt <- na_to_zeros(dt,c(
    "diameter",
    "max_intensity",
    "min_intensity",
    "mean_intensity",
    "nodule_pred",
    "diameter_patches",
    "score_patches")
  )
  
  final_df = dt[,.(total_nodules_unet=sum(!is.na(x)), 
                   total_nodules_patches = sum(!is.na(x_patches)),
                   big_nodules_unet = sum(diameter > 20,na.rm=T),
                   big_nodules_patches = sum(diameter_patches > 20,na.rm = T),
                   max_diameter = max(diameter,na.rm = T),
                   max_diameter_patches = max(diameter_patches,na.rm = T),
                   num_slices_unet = uniqueN(nslice[!is.na(x)]),
                   num_slices_patches = uniqueN(nslice[!is.na(x_patches)]),
                   nodules_per_slice_unet = sum(!is.na(x))/uniqueN(nslice[!is.na(x)]),
                   nodules_per_slice_patches = sum(!is.na(x_patches))/uniqueN(nslice[!is.na(x_patches)]),
                   max_intensity = max(max_intensity, na.rm=T), 
                   max_mean_intensity = max(mean_intensity, na.rm=T),
                   min_intensity = min(min_intensity,na.rm=T),
                   max_score = max(nodule_pred, na.rm=T),
                   mean_score = mean(nodule_pred),
                   max_score_patches = max(score_patches,na.rm=T),
                   mean_score_patches = mean(score_patches,na.rm=T)
                   
  ),
  by=.(patientid)]
  
  final_df[!is.finite(max_intensity), max_intensity:=0]
  final_df[!is.finite(min_intensity), min_intensity:=0]
  final_df[!is.finite(max_mean_intensity), max_mean_intensity:=0]
  final_df[is.na(final_df)] <- 0
  final_df[max_score < 0,max_score := 0]
  
  # Computing if the patient has consecutive nodules 
  dt_2d <- dt[
    nodule_pred > 0.2 & max_intensity > 0.96,
    .(patientid,nslice,x,y)]
  dt_2d_bis <- dt[
    nodule_pred > 0.2 & max_intensity > 0.96,
    .(patientid,nslice2 = nslice,x2 = x,y2 = y)]
  dt_3d <- merge(dt_2d,dt_2d_bis,all.x = T, by = "patientid",allow.cartesian = TRUE)
  dt_3d <- dt_3d[nslice2 > nslice]
  setkey(dt_3d,patientid,nslice,nslice2)
  dt_3d <- dt_3d[,.SD[1],c("patientid","nslice")]
  dt_3d <- dt_3d[nslice2-nslice < 3]
  dt_3d[,d_nodule := abs(x-x2)+abs(y-y2) < 10]
  nodules_consec <- dt_3d[,.(consec_nods = sum(d_nodule)),patientid]
  
  final_df <- merge(final_df,nodules_consec,all.x = T, by = "patientid")
  final_df[is.na(consec_nods),consec_nods := 0]
  
  
  # Computing if the patient has consecutive nodules of patches
  dt_2d <- dt[!is.na(x_patches),.(patientid,nslice,x = x_patches,y = y_patches)]
  dt_2d_bis <- dt[!is.na(x_patches),.(patientid,nslice2 = nslice,x2=x_patches,y2=y_patches)]
  dt_3d <- merge(dt_2d,dt_2d_bis,all.x=T,by="patientid",allow.cartesian = TRUE)
  dt_3d <- dt_3d[nslice2 > nslice]
  setkey(dt_3d,patientid,nslice,nslice2)
  dt_3d <- dt_3d[,.SD[1],c("patientid","nslice")]
  dt_3d <- dt_3d[nslice2-nslice <3]
  dt_3d[,d_nodule := abs(x-x2) + abs(y-y2) < 10]
  nodules_patches_consec <- dt_3d[,.(consec_nods_patches = sum(d_nodule)),patientid]
  
  final_df <- merge(final_df,nodules_patches_consec,all.x=T,by="patientid")
  final_df[is.na(consec_nods_patches),consec_nods_patches:=0]
  
  # Computing the variables of the nodules with highter score for patient
  dt[,`:=`(
    max_score = max(nodule_pred,na.rm=T),
    max_score_patches=max(score_patches,na.rm=T)),patientid]
  max_score <- dt[
    (max_score == nodule_pred & max_score > 0),
    .(patientid,
      diameter_nodule = diameter,
      max_intensity_nodule = max_intensity,
      nslice_nodule = nslice,
      mean_intensity_nodule = mean_intensity)
    ]
  
  max_score_nodule <- dt[
    max_score_patches == score_patches & max_score_patches > 0,
    .(patientid,
      nslice_nodule_patch = nslice,
      diameter_nodule_patch = diameter_patches)
    ]
  max_score_nodule <- max_score_nodule[,.SD[1],patientid]
  final_df <- merge(final_df,max_score,all.x = T, by="patientid")
  final_df <- merge(final_df,max_score_nodule,all.x=T,by = "patientid")
  final_df <- na_to_zeros(
    final_df,
    c("diameter_nodule",
      "max_intensity_nodule",
      "nslice_nodule",
      "mean_intensity_nodule",
      "nslice_nodule_patch",
      "diameter_nodule_patch")
  )
  return(final_df)
}
na_to_zeros <- function(dt,name_vars) {
  for(name_var in name_vars) {
    setnames(dt,name_var,"id")
    if(nrow(dt[is.na(id)]) > 0) {
      dt[is.na(id),id := 0]
    }
    setnames(dt,"id",name_var)
  }
  return(dt)
}