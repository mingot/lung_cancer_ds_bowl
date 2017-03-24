import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy import stats
from PIL import Image

# Define folder locations
INPUT_FOLDER = '/Users/rdg/Documents/my_projects/DSB17/emphysema/'

PATCH_FOLDER = os.path.join(INPUT_FOLDER, 'patches/')
PATCH_LABEL_FILENAME = os.path.join(INPUT_FOLDER, 'patch_labels.csv')
SLICE_INFO_FILENAME = os.path.join(INPUT_FOLDER, 'slice_labels.csv')


threshold = -600


def get_emphysema_predictors(img, mask):

    pix_1d = np.ndarray.flatten(img)
    mask_1d = np.ndarray.flatten(mask)
    pix_lung = pix_1d[mask_1d > 0]
    gated_pix_lung = pix_lung[pix_lung < threshold]
    gated_skewness = stats.skew(gated_pix_lung)
    gated_kurtosis = stats.kurtosis(gated_pix_lung)

    return gated_skewness, gated_kurtosis


def compute_emphysema_probability(img, mask):

    with open('utils/emphysema_models/neural_net_model.sav', 'rb') as fid:
        clf = pickle.load(fid)
    gated_skewness, gated_kurtosis = get_emphysema_predictors(img, mask)
    # X_new_preds = clf.predict([gated_skewness, gated_kurtosis])
    probability = clf.predict_proba([gated_skewness, gated_kurtosis])

    return probability[0, 1], gated_skewness, gated_kurtosis

