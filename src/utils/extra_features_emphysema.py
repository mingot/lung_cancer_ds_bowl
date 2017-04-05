import os
import csv
import pickle
import numpy as np
from scipy.stats import entropy
import pywt
from plotting import multiplot_single_image


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

COMMON_SPACING = [2, 0.7, 0.7]


def __entropy2__(labels):
    """ Computes entropy of label distribution. """

    min_val = np.min(labels)
    max_val = np.max(labels)
    data_array = labels.flatten()

    hist, bin_edges = np.histogram(data_array, bins=range(int(min_val), int(max_val), 1))
    ent = entropy(hist)

    return ent


def __compute_emphysema_entropy__(img, mask):
    img_masked = img[mask == 1]
    data_array = img_masked.flatten()
    ent = __entropy2__(data_array)

    return ent


def compute_emphysema_amfm(img, mask):
    """automated texture-based adaptive multiple feature method"""

    ent = __compute_emphysema_entropy__(img, mask)

    img_masked = img * mask
    coeffs = pywt.wavedecn(img_masked, 'db1')

    c0 = np.log(np.mean(np.abs(coeffs[0].flatten())))
    c1 = np.log(np.mean(np.abs(coeffs[1]["ddd"].flatten())))
    c2 = np.log(np.mean(np.abs(coeffs[2]["ddd"].flatten())))
    c3 = np.log(np.mean(np.abs(coeffs[3]["ddd"].flatten())))
    c4 = np.log(np.mean(np.abs(coeffs[4]["ddd"].flatten())))
    c5 = np.log(np.mean(np.abs(coeffs[5]["ddd"].flatten())))
    c6 = np.log(np.mean(np.abs(coeffs[6]["ddd"].flatten())))
    c7 = np.log(np.mean(np.abs(coeffs[7]["ddd"].flatten())))

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([c0, c1, c2, c3, c4, c5, c6, c7])
    slope = (len(x)*np.sum(x*y)-np.sum(x)*np.sum(y))/(len(x)*np.sum(x*x)-np.sum(x)*np.sum(x))

    return ent, slope


def compute_emphysema_mld(img, mask):
    """Mean Lung Density Method"""

    img_masked = img[mask == 1]
    data_array = img_masked.flatten()
    voxel_volume_l = COMMON_SPACING[0] * COMMON_SPACING[1] * COMMON_SPACING[2]
    mld = np.mean(data_array) / voxel_volume_l

    return mld


def compute_emphysema_hist(img, mask):
    """lowest fifth percentile of the density histogram method"""
    img_masked = img[mask == 1]
    min_val = np.min(img_masked)
    max_val = np.max(img_masked)
    data_array = img_masked.flatten()

    hist, bin_edges = np.histogram(data_array, bins=range(int(min_val), int(max_val), 1))
    pixel_counts = np.sum(hist)
    accumulated = np.cumsum(hist)/float(pixel_counts)

    index = 0
    for x in accumulated:
        if x > 0.2:
            break
        index += 1

    fifth_percentile = bin_edges[index]

    return fifth_percentile


def compute_emphysema_crap(img, mask):
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

    # feat1, feat2 = compute_emphysema_crap(img, mask)
    feat3 = compute_emphysema_mld(img, mask)
    feat4 = compute_emphysema_hist(img, mask)

    temp = np.zeros((1, 2), dtype=np.float)
    temp[0, 0] = feat3
    temp[0, 1] = feat4
    probability = clf.predict_proba(temp)

    # more extra features
    feat5, feat6 = compute_emphysema_amfm(img, mask)

    return probability[0, 1], feat3, feat4, feat5, feat6


def process_patient_file(patient_name):
    file_name = os.path.join(path_preprocessed, patient_name)
    patient_id = patient_name[4:-4]

    print('Processing patient {}'.format(patient_id))

    saved_data = np.load(file_name)
    loaded_stack = saved_data['arr_0']
    img = loaded_stack[0, :, :, :]
    mask = loaded_stack[1, :, :, :]
    p, f1, f2, f3, f4 = compute_emphysema_probability(img, mask)
    new_row = [patient_id, p, f1, f2, f3, f4]
    print(new_row)
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

