# Example of csv feature augmentation
# needed columns: patientid (with .npz), nslice, x, y, diameter

import os

# Join src path in the dsb repo
PATH_LUNG = os.environ.get('LUNG_PATH')
os.chdir(PATH_LUNG)

# my crappy functions
# import sp_lib.model as splm
from nodules_aggregator.extend_nodules import process_pipeline_csv
import pandas as pd

# example with patients:
# dsb_0700108170c91ea2219006e9484999ef.npz
# dsb_6d3b16f2e60c3a1a4246f340dba73676.npz
# csv_in = PATH_LUNG + 'personal/noduls_patches_v05_backup3_tiny.csv'
# csv_out = PATH_LUNG + 'personal/noduls_patches_v05_backup3_tiny_out.csv'
# patient_path = PATH_LUNG + 'data/preprocessed5_sample'
csv_out = 'data/tiny_dl_example_augmented.csv'
csv_in = 'data/tiny_dl_example.csv'
# csv_in = 'data/output_model_teixi_luna2.csv'
# csv_in = 'data/noduls_patches_v05_backup3_sample.csv'
# patient_path = "/home/sergi/all/devel/big/lung_cancer_ds_bowl/preprocessed5/"
patient_path = "/media/sergi/Seagate Expansion Drive/preprocessed5/"

# read csv from deep learning and write new csv
process_pipeline_csv(
    csv_in=csv_in, 
    patient_path=patient_path, 
    csv_out=csv_out, 
    patient_colname='patientid',
    nCores=2)
    
df_augmented = pd.read_csv(csv_out)
df_augmented.head()

