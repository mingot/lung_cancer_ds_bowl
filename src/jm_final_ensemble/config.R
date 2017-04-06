path_repo <<- "~/lung_cancer_ds_bowl/"
path_dsb <<- "/home/shared/output/"

# file locations
ANNOTATIONS_FILE = paste0(path_repo,"/data/stage1_labels.csv")  # JUST FOR TRAINING
SUBMISSIONS_FILE = paste0(path_repo,"/data/stage1_sample_submission.csv")
DL1_FILE = paste0(path_dsb,"resnet/nodules_patches_dl1_v11.csv")
DL2_FILE = paste0(path_dsb,"resnet/nodules_patches_hardnegative_v03.csv")
NODULES_AGGREGATED_FILE = paste0(path_dsb,"resnet/nodules_patches_dl1_v11_score07_noduleaggr_augmented.csv")
DL3_FILE = paste0(path_dsb,"resnet/nodules_patches_dl3_v02.csv")
DL3_VALIDATION_FILE = paste0(path_repo, 'data/stage1_validation.csv')
FINAL_MODEL_01 = "final_model_01.rda"
FINAL_MODEL_02 = "final_model_02.rda"
SUBMISSION_OUTPUT01 = ""
SUBMISSION_OUTPUT02 = ""


