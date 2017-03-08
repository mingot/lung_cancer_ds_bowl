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
sys.path.append(os.path.dirname(__file__) + "../../")
from networks.sample_cnn import Sample2DCNNNetworkArchitecture
from networks.sample_resnet import ResnetBuilder
from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
#K.set_image_dim_ordering('th')


# PARAMETERS
NUM_EPOCHS = 100
BATCH_SIZE = 100
USE_EXISTING = False  # load previous model to continue training


## paths
model_path  = '/mnt/hd2/models'
custom_dataset_path = '/mnt/hd2/custom_dataset_jc'
logs_path = '/mnt/hd2/logs/slice_%s' % str(int(time()))

def get_data_from_file(filename):
    aux = np.load(filename)
    return np.expand_dims(np.asarray(aux['X']),axis=1), aux['Y']

test_files = [custom_dataset_path + x for x in os.listdir(custom_dataset_path) if 'custom_dataset_test_subsample_' in x]
X_test, Y_test = get_data_from_file(test_files[0])

train_files = [custom_dataset_path + x for x in os.listdir(custom_dataset_path) if 'custom_dataset_train_subsample_' in x][:2]
print(train_files)


if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard


print 'creating model...'
model = ResnetBuilder().build_resnet_18((1,512,512),1)
model.compile(optimizer=Adam(lr=.5e-2), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  # metric which will be used is defined here

if USE_EXISTING:
    print 'loading model...'
    model.load_weights(model_path + 'fg_sampleresnet18_v0.hdf5')


## Load LUNA dataset
# Load dataset
normalize = lambda x, mean_, std_: (x - mean_) / std_
X_test = normalize(X_test, X_test.mean(), X_test.std())

'''
X_train, Y_train = get_data_from_file(train_files[0])
X_train = normalize(X_train, X_test.mean(), X_test.std())
for i in range(NUM_EPOCHS):
    model.fit(X_train, Y_train, class_weight = {0:0.1,1:0.9}, verbose=1, nb_epoch=25, batch_size=20, validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])
    model.save(model_path + 'jc_samplecnn_v0.hdf5')
'''

#print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    random.shuffle(train_files)
    for file in train_files:
        X_train, Y_train = get_data_from_file(file)
        X_train = normalize(X_train, X_test.mean(), X_test.std())
        print("Ys labeled as 1s: " + str(Y_train.sum()))
        model.fit(X_train, Y_train, class_weight = {0:1.,1:10.}, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])
        model.save(model_path + 'fg_sampleresnet18_v0.hdf5')
        del X_train
        del Y_train
