#   JC_PREPARE_SLICES_DATA

import copy
import os
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from numpy.random import binomial

K.set_image_dim_ordering('th')

## paths
TEST_CASES = 20
CASES_PER_FILE = 1000

wp = os.environ['LUNG_PATH']
model_path  = wp + 'models/'
input_path = '/mnt/hd2/preprocessed4'
#input_path = '/home/jose/kaggle/cfis/lung_cancer_ds_bowl/data/sample_data'

custom_dataset_path = wp + 'src/jc_dl/experiments/slice_classification/new_dataset/'
try:
    os.makedirs(custom_dataset_path)
except OSError as err:
    if err.errno!=17:
        raise

normalize = lambda x: (x - np.mean(x))/np.std(x)

def get_slices_patient( filelist,
                        ignore_extreme_slices = 10,
                        skip_no_mask = False,
                        distance_between_slices = 5,
                        p_keep_no_nodule_slice = 0.2):
    X, Y = [], []
    for cur_i, filename in enumerate(filelist):
        print("- %d out of %d    - %d" % (cur_i, len(filelist), len(X)))
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

            # othewise I take it
            else:
                # Discard if bad segmentation
                voxel_volume_l = 2*0.7*0.7/(1000000.0)
                lung_volume_l = np.sum(lung_mask)*voxel_volume_l
                if lung_volume_l < 0.02 or lung_volume_l > 0.1:
                    print("bad lung segmentation")
                    continue  # skip slices with bad lung segmentation


                # A AFEGIR: nodules out of lungs
                if np.any(np.logical_and(nodules_mask, 0 == lung_mask)):
                    print 'nodules out of lungs for %s at %d' % (filename, j)
                    continue

                #nodule_slices.append(j)
                last_slice = j
                X.append(np.array(lung_image))
                Y.append(1)


        if len(X) >= CASES_PER_FILE:
            yield np.array(X[:CASES_PER_FILE]), np.array(Y[:CASES_PER_FILE])
            X, Y = list(X[CASES_PER_FILE:]), list(Y[CASES_PER_FILE:])
        del b
    yield X, Y

import random
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]
random.shuffle(file_list)

# This way you can load more images from test and reduze train to collect a subsample

current_i = 0
for test_dataset_slices in get_slices_patient(file_list[-TEST_CASES:], p_keep_no_nodule_slice = 0.05):
    if len(test_dataset_slices[0]) == 0:
        continue
    current_i += 1
    X_test  = test_dataset_slices[0]
    Y_test  = test_dataset_slices[1]
    np.savez(custom_dataset_path + 'custom_dataset_test_subsample_%d' % current_i, X=X_test, Y=Y_test)


current_i = 0
for train_dataset_slices in get_slices_patient(file_list[:-TEST_CASES], p_keep_no_nodule_slice = 0.05):
    if len(train_dataset_slices[0]) == 0:
        continue
    current_i += 1
    X_train = train_dataset_slices[0]
    Y_train = train_dataset_slices[1]
    np.savez(custom_dataset_path + 'custom_dataset_train_subsample_%d' % current_i, X=X_train, Y=Y_train)
