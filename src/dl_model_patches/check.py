import os
import numpy as np
from dl_model_patches.common import load_patient, add_stats
### Quality checks for ROIs detection
wp = os.environ['LUNG_PATH']
INPUT_PATH = wp + 'data/preprocessed5_sample'
# INPUT_PATH = wp + 'data/preprocessed5_sample_watershed'
# INPUT_PATH = wp + 'data/preprocessed5_sample_th2'
file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)]

total_stats = {}
for filename in file_list:
    print "Loading patient: " + filename
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = load_patient(patient_data=patient_data, patient_nodules_df=None,
                                     discard_empty_nodules=True, output_rois=True,
                                     debug=True, thickness=1)
    print stats
    total_stats = add_stats(stats, total_stats)
    print "TOTAL STATS:", filename.split('/')[-1], total_stats
    break
