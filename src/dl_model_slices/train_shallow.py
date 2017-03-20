import os
import csv
import sys
import copy
import math
import random
import os.path
import numpy as np
from time import time
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam
from numpy.random import binomial
from sklearn.metrics import roc_auc_score
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/")
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard

NUM_EPOCHS = 100
BATCH_SIZE = 20
USE_EXISTING = True
TEST_CASES = 200
train_input_path = '/mnt/hd2/preprocessed5/'
model_path = '/mnt/hd2/models/'
logs_path = '/mnt/hd2/logs/slice_%s' % str(int(time()))
train_files = [os.path.join(train_input_path,g) for g in os.listdir(train_input_path) if g.startswith('luna_')][TEST_CASES:]
#test_files = [os.path.join(test_input_path,g) for g in os.listdir(test_input_path) if g.startswith('luna_')][:TEST_CASES]
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False) # replace keras.callbacks.TensorBoard
model = ResnetBuilder().build_resnet_18((512,1,512),1)
model.compile(optimizer=Adam(lr=0.5e-4), loss='binary_crossentropy', metrics=['accuracy'])

if USE_EXISTING: model.load_weights(model_path + 'fg_sampleresnet18_v0.hdf5')

def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    # hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

for i in range(NUM_EPOCHS):
    random.shuffle(train_files)
    for j,file in enumerate(train_files):
        myfile = np.load(os.path.join(input_path, filename))['arr_0']
        X = myfile[0,:,:,:]
        X = X.reshape([ X.shape[0], 1, X.shape[1], X.shape[2] ])
        X = normalize(np.array(X))
        Y = np.zeros([X.shape[0]], dtype=float)
        if myfile.shape[0]==3:
            for j in range(X.shape[0]):
                Y[j] = int(myfile[2,j,:,:].sum() != 0)
        model.fit(X, Y, class_weight = None, verbose=2, nb_epoch=10, batch_size=BATCH_SIZE, validation_data=None, shuffle=True, callbacks=[tb])
        model.save(model_path + 'fg_sampleresnet18_v0.hdf5')
        del X_train
        del Y_train
