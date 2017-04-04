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
import os
import random
import logging
import multiprocessing
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from dl_model_patches import  common
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

OUTPUT_MODEL = wp + 'models/jc_simplepos_patches_train_v02.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % str('simplepos_v02')

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False) 
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)


## TRAINING -----------------------------------------------------------------


def chunks_multichannel(X, y, batch_size=32, augmentation_times=4, thickness=0, is_training=True):
    """
    Batches generator for keras fit_generator. Returns batches of patches 40x40px
     - augmentation_times: number of time to return the data augmented
     - concurrent_patients: number of patients to load at the same time to add diversity
     - thickness: number of slices up and down to add as a channel to the patch
    """
    y = np.expand_dims(y, axis=1)
    while 1:
        # downsample negatives (reduce 90%)
        if is_training:
            len1 = int(0.25*batch_size)
            idx_1 = [i for i in range(len(y)) if y[i]==1]
            idx_1 = random.sample(idx_1, len1)
            idx_0 = [i for i in range(len(y)) if y[i]==0]
            idx_0 = random.sample(idx_0, batch_size - len1)
            selected_samples = idx_0 + idx_1
            random.shuffle(selected_samples)
        else:
            selected_samples = range(len(y))

        #selected_samples  = [i for i in range(len(y_orig)) if y_orig[i]==1 or random.randint(0,9)==0]
        X = X[selected_samples]
        y = y[selected_samples]
        logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y), len(y)-sum(y)))

        flip = np.random.randint(2, size = (len(X),2)) * 2 -1
        a = np.array([X[i][0][:,::flip[i][0],::flip[i][1]] for i in range(len(X))]) # This one implements flips horizontal and vertical... simple data augmentation. THe flip array contains 1s and -1s meaning flip or o flip
        b = np.array([X[i][1] for i in range(len(X))])
        yield [a,b], y

PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'

# LOADING PATCHES FROM DISK
logging.info("Loading training and test sets")
x_train = np.load(os.path.join(PATCHES_PATH, 'x_train_dl_simplepos_0.npz'))['arr_0']
y_train = np.load(os.path.join(PATCHES_PATH, 'y_train_dl_simplepos_0.npz'))['arr_0']
x_test = np.load(os.path.join(PATCHES_PATH, 'x_test_dl_simplepos_0.npz'))['arr_0']
y_test = np.load(os.path.join(PATCHES_PATH, 'y_test_dl_simplepos_0.npz'))['arr_0']
logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))

N_extra_features = len(x_train[0][1])


from dl_networks.simplepos_resnet import simpleposResnet
model = simpleposResnet().get_posResnet((3,40,40),[N_extra_features])
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# logging.info('Loading exiting model...')
# model.load_weights(OUTPUT_MODEL)


model.fit_generator(
                    generator=chunks_multichannel(x_train, y_train, batch_size=32, thickness=2),
                    samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=3000,
                    verbose=1,
                    #class_weight={0:1., 1:4.},
                    callbacks=[tb, model_checkpoint],
                    validation_data=chunks_multichannel(x_test, y_test, batch_size=32, thickness=2),  # TODO: is_training=False
                    nb_val_samples=1000,
                    max_q_size=64,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers

