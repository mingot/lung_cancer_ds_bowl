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
OUTPUT_DL1 = wp + 'output/nodules_patches_dl1_v19_luna.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
INPUT_PATH = '/mnt/hd2/preprocessed5/' # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna/' # VALIDATION_PATH = wp + 'data/preprocessed5_sample'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'

OUTPUT_MODEL =  wp + 'models/jm_patches_hardnegative_v04.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % 'hn_v04' #str(int(time()))
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

# # luna annotated samples (do not train over the samples not annotated)
# luna_df = pd.read_csv(LUNA_ANNOTATIONS)
# annotated = list(set(['luna_%s.npz' % p.split('.')[-1] for p in luna_df['seriesuid']]))
#
# # filter TP and FP of the suggested by DL1
# SCORE_TH = 0.5
# nodules_df = pd.read_csv(OUTPUT_DL1)
# nodules_df = nodules_df[(nodules_df['score'] > SCORE_TH) | (nodules_df['label']==1)]
# nodules_df['nslice'] = nodules_df['nslice'].astype(int)
# logging.info("Shape nodules df: %s" % str(nodules_df.shape))
#
#
# # Construction of training and testsets
# filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(INPUT_PATH)]
# filenames_test = [os.path.join(VALIDATION_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(VALIDATION_PATH)]
#
#
# def __load_and_store(filename):
#     patient_data = np.load(filename)['arr_0']
#     ndf = nodules_df[nodules_df['patientid']==filename.split('/')[-1]]
#     X, y, rois, stats = common.load_patient(patient_data, ndf, output_rois=True, thickness=1)
#     logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
#     return X, y, stats
#
#
# common.multiproc_crop_generator(filenames_train,
#                                 os.path.join(PATCHES_PATH,'dl2_v04_x_train.npz'),
#                                 os.path.join(PATCHES_PATH,'dl2_v04_y_train.npz'),
#                                 __load_and_store,
#                                 parallel=True)
#
#
# common.multiproc_crop_generator(filenames_test,
#                                 os.path.join(PATCHES_PATH,'dl2_v04_x_test.npz'),
#                                 os.path.join(PATCHES_PATH,'dl2_v04_y_test.npz'),
#                                 __load_and_store,
#                                 parallel=True)


### TRAINING -------------------------------------------------------------------------------------------------------


# Data augmentation generator
train_datagen = ImageDataGenerator(dim_ordering="th",rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                                   horizontal_flip=True, vertical_flip=True)
test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure


def chunk_generator(X, y, thickness=0, batch_size=32, is_training=True):
    while 1:
        prct_pop = 0.3 if is_training else 1  # of all the training set, how much we keep
        prct1 = 0.2  # % of 1's
        idx_1 = [i for i in range(len(y)) if y[i]==1]
        idx_1 = random.sample(idx_1, int(prct_pop*len(idx_1)))
        idx_0 = [i for i in range(len(y)) if y[i]==0]
        idx_0 = random.sample(idx_0, int(len(idx_1)/prct1))
        selected_samples = idx_0 + idx_1
        random.shuffle(selected_samples)
        logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y[selected_samples]), len(y[selected_samples])-sum(y[selected_samples])))

        # generator: if testing, do not augment data
        data_generator = train_datagen if is_training else test_datagen

        i, good = 0, 0
        lenX = len(selected_samples)
        for X_batch, y_batch in data_generator.flow(X[selected_samples], y[selected_samples], batch_size=batch_size, shuffle=is_training):
            i += 1
            if good*batch_size > lenX*2 or i>100:  # stop when we have augmented enough the batch
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                continue
            good += 1
            yield X_batch, y_batch


# LOADING PATCHES FROM DISK
# Teoria TRAIN: Total time: 1040.66, total patients:576, stats: {'fp': 13182, 'fn': 22, 'tp': 2023}
# Teoria TEST: Total time: 35.66, total patients:18, stats: {'fp': 400, 'fn': 3, 'tp': 71}
logging.info("Loading training and test sets")
x_train = np.load(os.path.join(PATCHES_PATH, 'dl2_v04_x_train.npz'))['arr_0']
y_train = np.load(os.path.join(PATCHES_PATH, 'dl2_v04_y_train.npz'))['arr_0']
y_train = np.expand_dims(y_train, axis=1)
x_test = np.load(os.path.join(PATCHES_PATH, 'dl2_v04_x_test.npz'))['arr_0']
y_test = np.load(os.path.join(PATCHES_PATH, 'dl2_v04_y_test.npz'))['arr_0']
y_test = np.expand_dims(y_test, axis=1)
logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))

#Load model
model = ResnetBuilder().build_resnet_50((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', save_best_only=True)
# logging.info('Loading exiting model...')
# model.load_weights(OUTPUT_MODEL)


model.fit_generator(generator=chunk_generator(x_train, y_train, batch_size=32, thickness=1),
                    samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=1600,
                    verbose=1,
                    callbacks=[tb, model_checkpoint],
                    validation_data=chunk_generator(x_test, y_test, batch_size=32, thickness=1, is_training=False),
                    nb_val_samples=32*30,
                    max_q_size=64,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers

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