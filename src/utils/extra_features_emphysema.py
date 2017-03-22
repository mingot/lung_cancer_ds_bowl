import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from PIL import Image

# Define folder locations
INPUT_FOLDER = '/Users/rdg/Documents/my_projects/DSB17/emphysema/'

PATCH_FOLDER = os.path.join(INPUT_FOLDER, 'patches/')
PATCH_LABEL_FILENAME = os.path.join(INPUT_FOLDER, 'patch_labels.csv')
SLICE_INFO_FILENAME = os.path.join(INPUT_FOLDER, 'slice_labels.csv')


threshold = -250


def get_emphysema_predictors(img, mask):

    pix_1d = np.ndarray.flatten(img)
    mask_1d = np.ndarray.flatten(mask)
    pix_lung = pix_1d[mask_1d == 1]
    gated_pix_lung = pix_lung[pix_lung < threshold]

    # the histogram of the data
    # plt.hist(pix_lung, 50, normed=1, facecolor='green', alpha=0.75)

    gated_skewness = stats.skew(gated_pix_lung)
    gated_kurtosis = stats.kurtosis(gated_pix_lung)

    return gated_skewness, gated_kurtosis


def compute_emphysema_probability(img, mask):

    probability = 0

    gated_skewness, gated_kurtosis = get_emphysema_predictors(img, mask)

    # TODO: Combine gated_skewness, gated_kurtosis to obtain an estimate of the probability of the emphysema

    return probability


if __name__ == "__main__":

    patient_files = os.listdir(PATCH_FOLDER)

    count = 0
    patch_dict = {}
    try:
        f = open(PATCH_LABEL_FILENAME)
        reader = csv.reader(f)
        for row in reader:
            patch_dict[count] = row
            count += 1
            name_sub = 'patch' + str(count) + '.tiff'
            files = [s for s in patient_files if name_sub in s]
            image_filename = os.path.join(PATCH_FOLDER, files[0])
            im = Image.open(image_filename)
            imarray = np.array(im)
            print('{}'.format(imarray.shape))
            # TODO: Aixo peta aqui...
            im.show()

        f.close()
    except Exception as e:
        print('The file {} could not be read.'.format(PATCH_LABEL_FILENAME))



    for patient_file in patient_files:
        try:
            print('TRYING SOMETHING.')
        except Exception as e:  # Some patients have no data, ignore them
            print('There was some problem reading patient {}. Ignoring and live goes on.'.format(patient_file))
            print('Exception', e)
            continue


