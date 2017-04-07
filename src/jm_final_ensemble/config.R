path_repo <<- "/Users/mingot/Projectes/kaggle/ds_bowl_lung/"
path_dsb <<- "/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/execute_model/"

# file locations
ANNOTATIONS_FILE = paste0(path_repo,"data/stage1_labels.csv")  # JUST FOR TRAINING
DL3_VALIDATION_FILE = paste0(path_repo, 'data/stage1_validation.csv')  # JUST FOR TRAINING
SUBMISSIONS_FILE = paste0(path_dsb, 'output_csv/todo_submission.csv')  # CONTAINS THE FILENAMES TO BE SUBMITTED
DL1_FILE = paste0(path_dsb,"output_csv/dl1.csv")
DL2_FILE = paste0(path_dsb,"output_csv/dl2.csv")
NODULES_EXTENDED_FILE = paste0(path_dsb,"output_csv/nodules_extended.csv")
DL3_FILE = paste0(path_dsb,"output_csv/dl3.csv")
FINAL_MODEL_01 = paste0(path_dsb, 'models/final_model_01.rda')
FINAL_MODEL_02 = paste0(path_dsb, 'models/final_model_02.rda')
SUBMISSION_OUTPUT01 = paste0(path_dsb, 'output_csv/submission01.csv')
SUBMISSION_OUTPUT02 = paste0(path_dsb, 'output_csv/submission02.csv')


