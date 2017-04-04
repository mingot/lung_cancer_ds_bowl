import os
import random
import logging
import multiprocessing
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from utils import plotting
from dl_model_simplepos_patches import  common
from sklearn import metrics
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard




# PATHS
wp = os.environ['LUNG_PATH']


INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'
NODULES_PATH = wp + 'data/luna/annotations.csv'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'
#PATCHES_PATH = '/home/jose/patches_temp'

OUTPUT_MODEL = wp + 'models/jc_simplepos_patches_train_v01.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % str('simplepos_v01')

#LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)



# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
#tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
#model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


### PATCHES GENERATION -----------------------------------------------------------------

## PATIENTS FILE LIST
patients_with_annotations = pd.read_csv(NODULES_PATH)  # filter patients with no annotations to avoid having to read them
patients_with_annotations = list(set(patients_with_annotations['seriesuid']))
patients_with_annotations = ["luna_%s.npz" % p.split('.')[-1] for p in patients_with_annotations]

filenames = os.listdir(INPUT_PATH)
filenames = [g for g in filenames if g.startswith('luna_')]
filenames_train = [os.path.join(INPUT_PATH, fp) for fp in filenames if fp in patients_with_annotations]
filenames_test = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH) if fp in patients_with_annotations]


def __load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = common.load_patient_3d_atlas(patient_data, 
                                                                 discard_empty_nodules=True, output_rois=True, debug=True, thickness=1,                                                               )
    logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
    return X, y, stats

common.multiproc_crop_generator(filenames_train,
                                os.path.join(PATCHES_PATH,'x_train_dl_simplepos_0.npz'),
                                os.path.join(PATCHES_PATH,'y_train_dl_simplepos_0.npz'),
                                __load_and_store, store=True)

common.multiproc_crop_generator(filenames_test,
                                 os.path.join(PATCHES_PATH,'x_test_dl_simplepos_0.npz'),
                                 os.path.join(PATCHES_PATH,'y_test_dl_simplepos_0.npz'),
                                 __load_and_store)
