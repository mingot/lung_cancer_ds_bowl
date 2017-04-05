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


INPUT_PATH = ''
PREPROCESSED_PATH = ''

## Preprocess data
logging.info('Preprocessing data ...')
cmnd = 'python 00_preprocess.py --input=/home/shared/data/stage1 --output=/mnt/hd2/preprocessed5/ --pipeline=dsb'
execute_command(cmnd)


OUTPUT_DL1 = ''
MODEL_DL1 = ''
OUTPUT_DL2 = ''
MODEL_DL2 = ''
OUTPUT_DL3 = ''
MODEL_DL3 = ''
INPUT_DL3 = ''
AGGREGATED_NODULES = ''

## Execute DL1
logging.info('Executign DL1 ...')
cmnd = 'python evaluate.py --input_folder=%s --input_model=%s --output_file=%s' % (PREPROCESSED_PATH, MODEL_DL1, OUTPUT_DL1)
execute_command(cmnd)

## Execute DL2
logging.info('Executign DL2 ...')
cmnd = 'python evaluate.py --input_folder=%s --input_model=%s --output_file=%s' % (PREPROCESSED_PATH, MODEL_DL2, OUTPUT_DL2)
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
cmnd = 'python evaluate.py --input_folder=%s --input_model=%s input_csv=%s --output_file=%s' % (PREPROCESSED_PATH, MODEL_DL3, INPUT_DL3, OUTPUT_DL3)
execute_command(cmnd)


## Execute emphysema

## (Gabriel) Aggregate nodules
logging.info('Executign nodules aggregation ...')
cmnd = 'python mergeNodules.py %s %s' % (INPUT_DL3, AGGREGATED_NODULES)
execute_command(cmnd)




## (Sergi) Extend nodules

## Execute final model 1

## Execute final model 2 (DL3)