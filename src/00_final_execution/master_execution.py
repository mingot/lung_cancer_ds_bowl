import sys
import os
import logging
import pandas as pd
from subprocess import check_output, CalledProcessError, STDOUT
import evaluate


### PARAMETERS
# BD = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/execute_model/'  # Base directory
# INPUT_PATH = BD + 'input_data/'
BD = '/home/shared/output/execution_test/'
INPUT_PATH = '/home/shared/data/sample_submission/'


# Creates folders if they do not exist
if not os.path.exists(BD + 'preprocessed_data'): os.makedirs(BD + 'preprocessed_data')
if not os.path.exists(BD + 'output_csv'): os.makedirs(BD + 'output_csv')
if not os.path.exists(BD + 'models'): os.makedirs(BD + 'models')


# TEMP files and directories
PREPROCESSED_PATH = BD + 'preprocessed_data/'  # /mnt/hd2/preprocessed5/
OUTPUT_DL1 = BD + 'output_csv/dl1.csv'
MODEL_DL1 = BD + 'models/jm_patches_train_v11.hdf5'
OUTPUT_DL2 = BD + 'output_csv/dl2.csv'
MODEL_DL2 = BD + 'models/jm_patches_hardnegative_v03.hdf5'
OUTPUT_DL3 = BD + 'output_csv/dl3.csv'
MODEL_DL3 = BD + 'models/jm_patches_malign_v03.hdf5'
INPUT_DL3 = BD + 'output_csv/dl12.csv'
AGGREGATED_NODULES = BD + 'output_csv/nodules_aggregated.csv'
EXTENDED_NODULES = BD + 'output_csv/nodules_extended.csv'
SUBMISSIONS_FILE = BD + 'output_csv/todo_submission.csv'


# Creates submission file
patients = os.listdir(INPUT_PATH)
with open(SUBMISSIONS_FILE,'w') as file:
    file.write('id,cancer\n')
    for patient in patients:
        file.write('%s,0.5\n' % patient)


# ## Preprocess data
# logging.info('Preprocessing data ...')
# from preprocess import preprocess_files
# patient_files = [os.path.join(INPUT_PATH, p) for p in os.listdir(INPUT_PATH)]
# preprocess_files(file_list=patient_files, output_folder=PREPROCESSED_PATH, pipeline='dsb')



file_list = [os.path.join(PREPROCESSED_PATH, fp) for fp in os.listdir(PREPROCESSED_PATH)]



# ## Execute DL1
# logging.info('Executign DL1 ...')
# evaluate.evaluate_model(file_list=file_list, model_path=MODEL_DL1, output_csv=OUTPUT_DL1)



## Execute DL2
logging.info('Executign DL2 ...')
evaluate.evaluate_model(file_list=file_list, model_path=MODEL_DL2, output_csv=OUTPUT_DL2)



## Execute DL3
logging.info("Loading DL1 and DL2 data frames...")
dl1_df = pd.read_csv(OUTPUT_DL1)
dl1_df = dl1_df[dl1_df['patientid'].str.startswith('dsb')]  # Filter DSB patients
dl2_df = pd.read_csv(OUTPUT_DL2)
dl3_df = pd.merge(dl1_df, dl2_df, on=['patientid','nslice','x','y','diameter'], how='inner', suffixes=('_dl1', '_dl2'))
dl3_df = dl3_df[((dl3_df['score_dl1'] + dl3_df['score_dl2'])/2 > 0.5) & (dl3_df['diameter']>7)]   # 12k candidates
# TODO: define 'score' column and remove 'score_dl1' i 'score_dl2'
dl3_df.to_csv(INPUT_DL3)

logging.info('Executign DL3 ...')
evaluate.evaluate_model(file_list=file_list, model_path=MODEL_DL3, output_csv=OUTPUT_DL3, nodules_df=dl3_df)



## (Gabriel) Aggregate nodules
THRESHOLD_CUT = 0.7
from merge_nodules import  merge_nodules_csv
logging.info('Executing nodules aggregation ...')
merge_nodules_csv(OUTPUT_DL1, AGGREGATED_NODULES, nodule_threshold=THRESHOLD_CUT)  # TODO: te mes sentit usar ja els HN?


## (Sergi) Extend nodules
import nodules_aggregator.extend_nodules as naen
logging.info('Executing nodules feature extraction ...')
naen.process_pipeline_csv(
    csv_in=AGGREGATED_NODULES,
    patient_path=PREPROCESSED_PATH,
    csv_out=EXTENDED_NODULES,
    patient_colname='patientid',
    dmin = 3, dmax = 100, # filtre de diametre
    compress={'hog':3, 'lbp':3, 'hu':2}, # quines features comprimir i amb quants pcs
    patient_inverted=[], #npz invertits
    nCores=4)



# ## Execute final R model
cmnd = 'Rscript final_minimal.R'
try:
    cmnd_output = check_output(cmnd, stderr=STDOUT, shell=True, universal_newlines=True)
except CalledProcessError as exc:
    logging.error(exc.output)
    sys.exit(1)
else:
    logging.info('Success! ' + cmnd_output)



