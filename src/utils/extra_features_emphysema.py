import os
import csv
import pickle
import numpy as np
from scipy import stats

SERVER = os.uname()[1] == 'ip-172-31-7-211'
MULTITHREAD = False

# Define execution location
if SERVER:
    path = '/home/shared/data/stage1'
    path_preprocessed = '/mnt/hd2/preprocessed5'
    output_file = '/home/ricard/var_emphysema_v05.csv'
else:
    path = '/Users/rdg/Documents/my_projects/DSB17/lung_cancer_ds_bowl/data/stage1'
    path_preprocessed = '/Users/rdg/Documents/my_projects/DSB17/lung_cancer_ds_bowl/data/stage1_proc'
    output_file = '/Users/rdg/Documents/my_projects/DSB17/lung_cancer_ds_bowl/data/stage1_proc/var_emphysema_v05.csv'

csvfile = open(output_file, 'wb')
csvwriter = csv.writer(csvfile, delimiter=',')


def get_emphysema_predictors(img, mask):
    # Threshold that gates the main lobe of the histogram
    threshold = -600

    pix_1d = np.ndarray.flatten(img)
    mask_1d = np.ndarray.flatten(mask)
    pix_lung = pix_1d[mask_1d > 0]
    gated_pix_lung = pix_lung[pix_lung < threshold]

    if len(gated_pix_lung) == 0:
        gated_skewness = 0
        gated_kurtosis = 3.0
    else:
        gated_skewness = stats.skew(gated_pix_lung)
        gated_kurtosis = stats.kurtosis(gated_pix_lung, fisher=False)

    return gated_skewness, gated_kurtosis


def compute_emphysema_probability(img, mask):

    with open('emphysema_models/neural_net_model.sav', 'rb') as fid:
        clf = pickle.load(fid)
    gated_skewness, gated_kurtosis = get_emphysema_predictors(img, mask)

    temp = np.zeros((1, 2), dtype=np.float)
    temp[0, 0] = gated_skewness
    temp[0, 1] = gated_kurtosis
    probability = clf.predict_proba(temp)

    return probability[0, 1], gated_skewness, gated_kurtosis


def process_patient_file(patient_name):
    file_name = os.path.join(path_preprocessed, patient_name)
    patient_id = patient_name[4:-4]

    print('Processing patient {}'.format(patient_id))

    saved_data = np.load(file_name)
    loaded_stack = saved_data['arr_0']
    img = loaded_stack[0, :, :, :]
    mask = loaded_stack[1, :, :, :]
    p, f1, f2 = compute_emphysema_probability(img, mask)
    new_row = [patient_id, p, f1, f2]
    print new_row
    csvwriter.writerow(new_row)


if __name__ == "__main__":
    print('server: {}'.format(SERVER))
    print('output_file: {}'.format(output_file))

    patient_files = os.listdir(path_preprocessed)
    patient_files = [f for f in patient_files if f.startswith('dsb')]
    patient_files = sorted(patient_files)

    for patient_file in patient_files:
        try:
            process_patient_file(patient_file)
        except Exception as e:  # Some patients have no data, ignore them
            print('There was some problem reading patient {}. Ignoring and live goes on.'.format(patient_file))
            print('Exception', e)
            continue

