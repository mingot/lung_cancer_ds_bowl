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
OUTPUT_DL1 = wp + 'output/nodules_patches_dl1_v11_rectif.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
OUTPUT_MODEL =  wp + 'models/jm_patches_hardnegative_v02.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
INPUT_PATH = '/mnt/hd2/preprocessed5/' # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna/' # VALIDATION_PATH = wp + 'data/preprocessed5_sample'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'
LOGS_PATH = wp + 'logs/%s' % 'hn_v02' #str(int(time()))
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
SCORE_TH = 0.7
nodules_df = pd.read_csv(OUTPUT_DL1)
nodules_df = nodules_df[(nodules_df['score'] > SCORE_TH) | (nodules_df['label']==1)]
#nodules_df['patientid'] = [f.split('/')[-1] for f in nodules_df['patientid']]
nodules_df['nslice'] = nodules_df['nslice'].astype(int)

# Construction of training and testsets
filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(INPUT_PATH)]
filenames_test = [os.path.join(VALIDATION_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(VALIDATION_PATH)]


def __load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    logging.info("Begin Patient %s" % filename.split('/')[-1])
    ndf = nodules_df[nodules_df['patientid']==filename.split('/')[-1]]
    X, y, rois, stats = common.load_patient(patient_data, ndf, output_rois=True, thickness=1)
    logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
    return X, y, stats


common.multiproc_crop_generator(filenames_train,
                                os.path.join(PATCHES_PATH,'x_train_dl2.npz'),
                                os.path.join(PATCHES_PATH,'y_train_dl2.npz'),
                                __load_and_store,
                                parallel=True)


common.multiproc_crop_generator(filenames_test,
                                os.path.join(PATCHES_PATH,'x_test_dl2.npz'),
                                os.path.join(PATCHES_PATH,'y_test_dl2.npz'),
                                __load_and_store,
                                parallel=True)


### TRAINING -------------------------------------------------------------------------------------------------------


# # Data augmentation generator
# train_datagen = ImageDataGenerator(dim_ordering="th", horizontal_flip=True, vertical_flip=True)
# test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure
#
#
# def chunk_generator(X_orig, y_orig, thickness=0, batch_size=32, is_training=True):
#     while 1:
#         logging.info("[TRAIN:%s] Loaded batch of patients with %d/%d positives" % (str(is_training), np.sum(y_orig), len(y_orig)))
#         idx_sel = [i for i in range(len(X_orig)) if y_orig[i]==1 or random.uniform(0,1) < 1.2*np.mean(y_orig)]
#         X = [X_orig[i] for i in idx_sel]
#         y = [y_orig[i] for i in idx_sel]
#         logging.info("Downsampled to %d/%d positives" % (np.sum(y), len(y)))
#
#         # convert to np array and add extra axis (needed for keras)
#         X, y = np.asarray(X), np.asarray(y)
#         y = np.expand_dims(y, axis=1)
#         if thickness==0:
#             X = np.expand_dims(X, axis=1)
#
#         # generator: if testing, do not augment data
#         data_generator = train_datagen if is_training else test_datagen
#
#         i, good = 0, 0
#         for X_batch, y_batch in data_generator.flow(X, y, batch_size=batch_size, shuffle=is_training):
#             i += 1
#             if good*batch_size > len(X)*2 or i>100:  # stop when we have augmented enough the batch
#                 break
#             if X_batch.shape[0] != batch_size:  # ensure correct batch size
#                 continue
#             good += 1
#             yield X_batch, y_batch
#
#
# # LOADING PATCHES FROM DISK
# # Teoria TRAIN: Total time: 1040.66, total patients:576, stats: {'fp': 13182, 'fn': 22, 'tp': 2023}
# # Teoria TEST: Total time: 35.66, total patients:18, stats: {'fp': 400, 'fn': 3, 'tp': 71}
# logging.info("Loading training and test sets")
# x_train = np.load(os.path.join(PATCHES_PATH, 'x_train_dl2.npz'))['arr_0']
# y_train = np.load(os.path.join(PATCHES_PATH, 'y_train_dl2.npz'))['arr_0']
# x_test = np.load(os.path.join(PATCHES_PATH, 'x_test_dl2.npz'))['arr_0']
# y_test = np.load(os.path.join(PATCHES_PATH, 'y_test_dl2.npz'))['arr_0']
# logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
# logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))
#
# # Load model
# model = ResnetBuilder().build_resnet_34((3,40,40),1)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
# # logging.info('Loading exiting model...')
# # model.load_weights(OUTPUT_MODEL)
#
#
# model.fit_generator(generator=chunk_generator(x_train, y_train, batch_size=32, thickness=1),
#                     samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
#                     nb_epoch=500,
#                     verbose=1,
#                     callbacks=[tb, model_checkpoint],
#                     validation_data=chunk_generator(x_test, y_test, batch_size=32, thickness=1, is_training=False),
#                     nb_val_samples=len(y_test),
#                     max_q_size=64,
#                     nb_worker=1)  # a locker is needed if increased the number of parallel workers

# # check generator
# for X,y in chunk_generator(filenames_train, nodules_df, batch_size=16):
#     print 'RESULT:', X.shape, y.shape



### EVALUATING -------------------------------------------------------------------------------------------------------

# nodules_df = pd.read_csv(OUTPUT_DL1)
# #nodules_df = nodules_df[nodules_df['score'] > SCORE_TH]  # TODO: this filter should include the TN through the label
# nodules_df['patientid'] = [f.split('/')[-1] for f in nodules_df['patientid']]  # TODO: remove when fixed the patient id without whole path
# nodules_df['nslice'] = nodules_df['nslice'].astype(int)
#
# # Construction of training and testsets
# filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(INPUT_PATH) and f in annotated]
# filenames_test = [os.path.join(VALIDATION_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(VALIDATION_PATH) and f in annotated]