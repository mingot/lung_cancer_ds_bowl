# Example of csv feature augmentation
# needed columns: patientid (with .npz), nslice, x, y, diameter

import os

# Join src path in the dsb repo
PATH_LUNG = os.environ.get('LUNG_PATH')
os.chdir(PATH_LUNG)

# my crappy functions
import sp_lib.model as splm
import pandas as pd

# example with patients:
# dsb_0700108170c91ea2219006e9484999ef.npz
# dsb_6d3b16f2e60c3a1a4246f340dba73676.npz
csv_in = 'data/tiny_dl_example.csv'
csv_out = 'data/tiny_dl_example_augmented.csv'
patient_path = "/home/sergi/all/devel/big/lung_cancer_ds_bowl/preprocessed5/"

# read csv from deep learning and write new csv
splm.process_pipeline_csv(
    csv_in=csv_in, 
    patient_path=patient_path, 
    csv_out=csv_out)

df_augmented = pd.read_csv(csv_out)
df_augmented.head()

