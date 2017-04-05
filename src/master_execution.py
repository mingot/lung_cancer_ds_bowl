import sys
import logging
import pandas as pd
from subprocess import check_output, CalledProcessError, STDOUT


def execute_command(cmnd):
    try:
        cmnd_output = check_output(cmnd, stderr=STDOUT, shell=True, universal_newlines=True)
    except CalledProcessError as exc:
        logging.error(exc.output)
        sys.exit(1)
    else:
        logging.info('Success! ' + cmnd_output)


INPUT_PATH = '' # /home/shared/data/stage1
PREPROCESSED_PATH = ''  # /mnt/hd2/preprocessed5/

## Preprocess data
logging.info('Preprocessing data ...')
cmnd = 'python 00_preprocess.py --input=%s --output=%s --pipeline=dsb' % (INPUT_PATH, PREPROCESSED_PATH)
execute_command(cmnd)


OUTPUT_DL1 = ''
MODEL_DL1 = ''
OUTPUT_DL2 = ''
MODEL_DL2 = ''
OUTPUT_DL3 = ''
MODEL_DL3 = ''
INPUT_DL3 = ''
AGGREGATED_NODULES = ''
EXTENDED_NODULES = ''

## Execute DL1
logging.info('Executign DL1 ...')
cmnd = 'python evaluate.py --input_path=%s --model=%s --output_file=%s' % (PREPROCESSED_PATH, MODEL_DL1, OUTPUT_DL1)
execute_command(cmnd)

## Execute DL2
logging.info('Executign DL2 ...')
cmnd = 'python evaluate.py --input_path=%s --model=%s --output_file=%s' % (PREPROCESSED_PATH, MODEL_DL2, OUTPUT_DL2)
execute_command(cmnd)

## Execute DL3
logging.info("Loading DL1 and DL2 data frames...")
dl1_df = pd.read_csv(OUTPUT_DL1)
dl1_df = dl1_df[dl1_df['patientid'].str.startswith('dsb')]  # Filter DSB patients
dl2_df = pd.read_csv(OUTPUT_DL2)
dl3_df = pd.merge(dl1_df, dl2_df, on=['patientid','nslice','x','y','diameter'], how='inner', suffixes=('_dl1', '_dl2'))
dl3_df = dl3_df[((dl3_df['score_dl1'] + dl3_df['score_dl2'])/2 > 0.5) & (dl3_df['diameter']>7)]   # 12k candidates
dl3_df.to_csv(INPUT_DL3)
logging.info('Executign DL3 ...')
cmnd = 'python evaluate.py --input_path=%s --model=%s input_csv=%s --output_file=%s' % (PREPROCESSED_PATH, MODEL_DL3, INPUT_DL3, OUTPUT_DL3)
execute_command(cmnd)


## TODO: Execute emphysema


## (Gabriel) Aggregate nodules
THRESHOLD_CUT = 0.7
logging.info('Executign nodules aggregation ...')
cmnd = 'python merge_nodules.py -t %s %s %s' % (str(THRESHOLD_CUT), OUTPUT_DL1, AGGREGATED_NODULES)  # TODO: te mes sentit usar ja els HN?
execute_command(cmnd)


## (Sergi) Extend nodules
import nodules_aggregator.extend_nodules as naen
naen.process_pipeline_csv(
    csv_in=AGGREGATED_NODULES,
    patient_path=PREPROCESSED_PATH,
    csv_out=EXTENDED_NODULES,
    patient_colname='patientid',
    dmin = 3, dmax = 100, # filtre de diametre
    compress={'hog':3, 'lbp':3, 'hu':2}, # quines features comprimir i amb quants pcs
    patient_inverted=[], #npz invertits
    nCores=4)


OUTPUT_TOTAL01 = ''
SUBMISSION = ''
CSV_FOLDER = ''  # contains all the intermediate csv files

## TODO: Execute final model 1
cmnd = 'R CMD --input_csv_folder=XX --output_total_model1=%s' % (CSV_FOLDER,OUTPUT_TOTAL01)


## TODO: Execute final model 2 (DL3)
cmnd = 'R CMD --input_dl3=XX --input_total_model1=XX --output_submission=XX' % (OUTPUT_TOTAL01, OUTPUT_DL3, SUBMISSION)
