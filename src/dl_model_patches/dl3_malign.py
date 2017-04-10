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
#LUNA_ANNOTATIONS = wp + 'data/luna/annotations.csv'
#OUTPUT_DL1 = wp + 'output/nodules_patches_dl1_v11.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
#OUTPUT_DL2 = wp + 'output/nodules_patches_hardnegative_v03.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
OUTPUT_DL12 = '/home/mingot/dl3/dl3_train_v05.csv'

DSB_VALIDATION = wp + 'data/stage1_validation_total.csv'
DSB_LABELS = wp + 'data/stage1_labels_total.csv'

INPUT_PATH = '/mnt/hd2/preprocessed5/' # INPUT_PATH = wp + 'data/preprocessed5_sample'
#VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna/' # VALIDATION_PATH = wp + 'data/preprocessed5_sample'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'

OUTPUT_MODEL =  wp + 'models/jm_patches_dl3_v05.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % 'malign_v05' #str(int(time()))
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

# labels
label_df = pd.read_csv(DSB_LABELS)
label_df['id'] = ["dsb_%s.npz" % p for p in list(label_df['id'])]

# Join DL1 and DL2 to filter TP and FP of the suggested by DL1
# dl1_df = pd.read_csv(OUTPUT_DL1)
# dl1_df = dl1_df[dl1_df['patientid'].str.startswith('dsb')]  # Filter DSB patients
# dl1_df = dl1_df[dl1_df.patientid.isin(list(label_df['id']))]  # keep only train patients
# dl2_df = pd.read_csv(OUTPUT_DL2)
# merge_df = pd.merge(dl1_df, dl2_df, on=['patientid','nslice','x','y','diameter'], how='inner', suffixes=('_dl1', '_dl2'))
# nodules_df = merge_df[((merge_df['score_dl1'] + merge_df['score_dl2'])/2 > 0.5) & (merge_df['diameter']>7)]   # 12k candidates
nodules_df = pd.read_csv(OUTPUT_DL12)
logging.info("DSB selected nodules shape: %s" % str(nodules_df.shape))


# Construction of training and testsets
validation_df = pd.read_csv(DSB_VALIDATION)
logging.info("DSB validation shape:%s" % str(validation_df.shape))
filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f not in list(validation_df['patientid'])]
filenames_test = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f in list(validation_df['patientid'])]

logging.info("Patients train:%d, test:%d" % (len(filenames_train), len(filenames_test)))

def __load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    patid = filename.split('/')[-1]
    ndf = nodules_df[nodules_df['patientid']==patid]
    X, y, rois, stats = common.load_patient(patient_data, ndf, output_rois=True, thickness=1)
    label = int(label_df[label_df['id']==patid]['cancer'])
    y = [label]*len(X)
    logging.info("Patient: %s, cancer:%d, stats: %s" % (patid, label, stats))
    return X, y, stats


common.multiproc_crop_generator(filenames_train,
                                os.path.join(PATCHES_PATH,'dl3_v05_x_train.npz'),
                                os.path.join(PATCHES_PATH,'dl3_v05_y_train.npz'),
                                __load_and_store,
                                parallel=True)


common.multiproc_crop_generator(filenames_test,
                                os.path.join(PATCHES_PATH,'dl3_v05_x_test.npz'),
                                os.path.join(PATCHES_PATH,'dl3_v05_y_test.npz'),
                                __load_and_store,
                                parallel=True)


### TRAINING -------------------------------------------------------------------------------------------------------



# # Data augmentation generator
# train_datagen = ImageDataGenerator(dim_ordering="th", horizontal_flip=True, vertical_flip=True, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1)
# test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure
#
#
# def chunks(X, y, batch_size=32, augmentation_times=4, thickness=0, is_training=True):
#     """
#     Batches generator for keras fit_generator. Returns batches of patches 40x40px
#      - augmentation_times: number of time to return the data augmented
#      - concurrent_patients: number of patients to load at the same time to add diversity
#      - thickness: number of slices up and down to add as a channel to the patch
#     """
#     while 1:
#         selected_samples = random.sample(range(len(y)), 1000)
#
#         #selected_samples  = [i for i in range(len(y_orig)) if y_orig[i]==1 or random.randint(0,9)==0]
#         logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y[selected_samples]), len(y[selected_samples])-sum(y[selected_samples])))
#
#         # generator: if testing, do not augment data
#         data_generator = train_datagen if is_training else test_datagen
#
#         i, good = 0, 0
#         lenX = len(selected_samples)
#         for X_batch, y_batch in data_generator.flow(X[selected_samples], y[selected_samples], batch_size=batch_size, shuffle=is_training):
#             i += 1
#             if good*batch_size > lenX*augmentation_times or i>100:  # stop when we have augmented enough the batch
#                 break
#             if X_batch.shape[0] != batch_size:  # ensure correct batch size
#                 continue
#             good += 1
#             yield X_batch, y_batch
#
#
# # LOADING PATCHES FROM DISK
# logging.info("Loading training and test sets")
# x_train = np.load(os.path.join(PATCHES_PATH, 'dl3_v04_x_train.npz'))['arr_0']
# y_train = np.load(os.path.join(PATCHES_PATH, 'dl3_v04_y_train.npz'))['arr_0']
# y_train = np.expand_dims(y_train, axis=1)
# x_test = np.load(os.path.join(PATCHES_PATH, 'dl3_v04_x_test.npz'))['arr_0']
# y_test = np.load(os.path.join(PATCHES_PATH, 'dl3_v04_y_test.npz'))['arr_0']
# y_test = np.expand_dims(y_test, axis=1)
# logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
# logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))
#
# # Load model
# model = ResnetBuilder().build_resnet_50((3,40,40),1)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='val_loss', save_best_only=True)
# # logging.info('Loading exiting model...')
# # model.load_weights(OUTPUT_MODEL)
#
#
# model.fit_generator(generator=chunks(x_train, y_train, batch_size=32, thickness=1),
#                     samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
#                     nb_epoch=1600,
#                     verbose=1,
#                     callbacks=[tb, model_checkpoint],
#                     validation_data=chunks(x_test, y_test, batch_size=32, thickness=1, is_training=False),
#                     nb_val_samples=2000,
#                     #initial_epoch=186, # TODO: RESUME PREVIOUS TRAINING
#                     max_q_size=32,
#                     nb_worker=1)  # a locker is needed if increased the number of parallel workers
