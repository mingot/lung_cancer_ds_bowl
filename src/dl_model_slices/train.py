import copy
import os
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from numpy.random import binomial
import random
import os
import sys
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import roc_auc_score
from dl_model_slices import dataset
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/")
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/jc_dl/")
#from networks.sample_cnn import Sample2DCNNNetworkArchitecture

from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
#K.set_image_dim_ordering('th')




# PARAMETERS
NUM_EPOCHS = 1000000
USE_EXISTING = True  # load previous model to continue training


## paths

if 'DL_ENV' in os.environ:
    if os.environ['DL_ENV'] == 'jose_local':
        model_path = '../models/'
        custom_dataset_path= './jc_dl/experiments/slice_classification/new_dataset/'
        logs_path = '../logs/slice_%s' % str(int(time()))
    else:
        raise 'No environment found'
else:
    model_path  = '/mnt/hd2/models/'
    custom_dataset_path = '/mnt/hd2/custom_dataset_jc/'
    logs_path = '/mnt/hd2/logs/slice_%s' % str(int(time()))

print("Logs in " + logs_path)

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False) # replace keras.callbacks.TensorBoard


print 'creating model...'
model = ResnetBuilder().build_resnet_50((512,1,512),1)
model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])

if USE_EXISTING:
    print 'loading model...'
    model.load_weights(model_path + 'jc_sampleresnet18_v0.hdf5')


train_gen, X_valid, Y_valid = dataset.get_dataset()

#print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    model.fit_generator(train_gen, 100, verbose=1, nb_epoch=1, validation_data=(X_valid, Y_valid), callbacks=[tb], class_weight={0:1.,1:5.}, nb_worker=1)
    model.save(model_path + 'jc_sampleresnet18_v0.hdf5')
