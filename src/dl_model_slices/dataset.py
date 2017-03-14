#   JC_PREPARE_SLICES_DATA

import copy
import os
import numpy as np
import math
from time import time
#from keras.optimizers import Adam
#from keras import backend as K
from numpy.random import binomial
import random
#K.set_image_dim_ordering('th')

## paths
TEST_CASES = 20
CASES_PER_META_BATCH = 500
CASES_PER_BATCH = 100
VALIDATION_SAMPLES = 320
#TEST_CASES = 20
#CASES_PER_META_BATCH = 20
#CASES_PER_BATCH = 10
#VALIDATION_SAMPLES = 5

if 'DL_ENV' in os.environ:
    if os.environ['DL_ENV'] == 'jose_local':
        model_path = '../models/'
        input_path = '../data/sample_data/'
        logs_path = '../logs/slice_%s' % str(int(time()))
    else:
        raise 'No environment found'
else:
    wp = os.environ['LUNG_PATH']
    model_path  = wp + 'models/'
    train_input_path = '/mnt/hd2/preprocessed5'
    val_input_path = '/mnt/hd2/preprocessed5_validation'


def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    # hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def get_slices_patient( filelist,
                        ignore_extreme_slices = 10,
                        skip_no_mask = False,
                        distance_between_slices = 5,
                        p_keep_no_nodule_slice = 0.2):
    X, Y = [], []
    while 1:

        random.shuffle(filelist)
        filename = filelist[0]

        b = np.load(os.path.join(input_path, filename))['arr_0']
        if b.shape[0]!=3:
            has_nodule = False
            slice_nodule = False
        else:
            has_nodule = True
            #print 'patient %s does not have the expected input shape' % (filename)
            #continue

        no_nodule_slices = []
        nodule_slices = []

        last_slice = -1e3  # big initialization

        for j in range(ignore_extreme_slices,b.shape[1]-ignore_extreme_slices):

            # discard consecutive slices
            if j<last_slice + distance_between_slices:
                continue

            lung_image = b[0,j,:,:]
            lung_mask = b[1,j,:,:]
            if has_nodule:
                nodules_mask = b[2,j,:,:]
                slice_nodule = nodules_mask.sum() != 0

            # if no nodule, I only keep the slice with some probability
            if not slice_nodule:
                if binomial(1, p_keep_no_nodule_slice) == 1:
                    #no_nodule_slices.append(j)
                    last_slice = j
                    X.append(np.array(lung_image))
                    Y.append(0)
            else:
                last_slice = j
                X.append(np.array(lung_image))
                Y.append(1)

        #print(len(X))
        if len(X) >= CASES_PER_META_BATCH:
            inds = range(len(X))
            np.random.shuffle(inds)
            X = np.array(X)[inds]
            Y = np.array(Y)[inds]
            dts = np.array(X[:CASES_PER_BATCH])
            yield normalize(dts.reshape([dts.shape[0], 1, dts.shape[1], dts.shape[2]])), np.array(Y[:CASES_PER_BATCH])

            X, Y = list(X[CASES_PER_BATCH:]), list(Y[CASES_PER_BATCH:])
        del b
        #yield X, Y

def get_dataset():
    train_file_list = [g for g in os.listdir(train_input_path) if g.startswith('luna_')]
    valid_file_list = [g for g in os.listdir(val_input_path) if g.startswith('luna_')][:TEST_CASES]

    X_valid, Y_valid = [],[]
    current_i = 0
    for train_dataset_slices in get_slices_patient(valid_file_list, p_keep_no_nodule_slice = 0.05):
        current_i += 1
        X_valid.append(train_dataset_slices[0])
        Y_valid.append(train_dataset_slices[1])
        if len(X_valid) * CASES_PER_BATCH >= VALIDATION_SAMPLES:
            X_valid = np.concatenate(X_valid)
            Y_valid = np.concatenate(Y_valid)
            break

    train_gen = get_slices_patient(train_file_list, p_keep_no_nodule_slice = 0.05)

    return train_gen, X_valid, Y_valid

if __name__ == '__main__':
    train_gen, X_valid, Y_valid = get_dataset()
    print("Done")
    for X_train, Y_train in train_gen:
        break