import os
import random
import logging
import pandas as pd
import numpy as np
from time import time
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from dl_networks.sample_resnet import ResnetBuilder
from dl_model_patches import common
from dl_utils.tb_callback import TensorBoard
from utils import plotting
import matplotlib.pyplot as plt
from skimage import transform


# PATHS
wp = os.environ['LUNG_PATH']
LUNA_ANNOTATIONS = wp + 'data/luna/annotations.csv'
OUTPUT_DL1 = wp + 'output/nodules_patches_dl1_v11.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
DSB_VALIDATION = wp + 'data/stage1_validation.csv'
INPUT_PATH = '/mnt/hd2/preprocessed5/' # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna/' # VALIDATION_PATH = wp + 'data/preprocessed5_sample'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'

OUTPUT_MODEL =  wp + 'models/jm_patches_malign_v01.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % 'malign_v03' #str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


### PATCHES GENERATION -----------------------------------------------------------------
# Load the output of DL-I and load just the 1's (TP or FN's) and the FP's for a given score
# to train DL-II

# luna annotated samples (do not train over the samples not annotated)
luna_df = pd.read_csv(LUNA_ANNOTATIONS)
annotated = list(set(['luna_%s.npz' % p.split('.')[-1] for p in luna_df['seriesuid']]))

# filter TP and FP of the suggested by DL1
SCORE_TH = 0.9
nodules_df = pd.read_csv(OUTPUT_DL1)
nodules_df = nodules_df[nodules_df['patientid'].str.startswith('dsb')]  # Filter DSB patients
nodules_df = nodules_df[(nodules_df['score'] > SCORE_TH) | (nodules_df['diameter']>10)]
nodules_df['nslice'] = nodules_df['nslice'].astype(int)
logging.info("DSB selected nodules shape: %s" % str(nodules_df.shape))


# Construction of training and testsets
validation_df = pd.read_csv(DSB_VALIDATION)
logging.info("DSB validation shape:%s" % str(validation_df.shape))
filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='dsb' and f in os.listdir(INPUT_PATH)]
filenames_test = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='dsb' and f in validation_df['patientid']]


def __load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    ndf = nodules_df[nodules_df['patientid']==filename.split('/')[-1]]
    X, y, rois, stats = common.load_patient(patient_data, ndf, output_rois=True, thickness=1)
    logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
    return X, y, stats


# common.multiproc_crop_generator(filenames_train,
#                                 os.path.join(PATCHES_PATH,'x_train_dl3.npz'),
#                                 os.path.join(PATCHES_PATH,'y_train_dl3.npz'),
#                                 __load_and_store,
#                                 parallel=True)
#
#
# common.multiproc_crop_generator(filenames_test,
#                                 os.path.join(PATCHES_PATH,'x_test_dl3.npz'),
#                                 os.path.join(PATCHES_PATH,'y_test_dl3.npz'),
#                                 __load_and_store,
#                                 parallel=True)

