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
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/")
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/jc_dl/")
#from networks.sample_cnn import Sample2DCNNNetworkArchitecture
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
#K.set_image_dim_ordering('th')


# PARAMETERS
NUM_EPOCHS = 100
BATCH_SIZE = 20
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

def get_data_from_file(filename):
    aux = np.load(filename)
    return np.expand_dims(np.asarray(aux['X']),axis=1), aux['Y']

test_files = [custom_dataset_path + x for x in os.listdir(custom_dataset_path) if 'custom_dataset_test_subsample_' in x]
X_test, Y_test = get_data_from_file(test_files[0])

train_files = [custom_dataset_path + x for x in os.listdir(custom_dataset_path) if 'custom_dataset_train_subsample_' in x]
print(train_files)


if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False) # replace keras.callbacks.TensorBoard


print 'creating model...'
model = ResnetBuilder().build_resnet_18((512,1,512),1)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  # metric which will be used is defined here

#if USE_EXISTING:
#    print 'loading model...'
#    model.load_weights(model_path + 'fg_sampleresnet18_v0.hdf5')


## Load LUNA dataset
# Load dataset
normalize = lambda x, mean_, std_: (x - mean_) / std_
#true_x_test_mean = X_test.mean()
#true_x_test_std = X_test.std()
#X_test = normalize(X_test,true_x_test_mean, true_x_test_std)


#print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    random.shuffle(train_files)
    for j,file in enumerate(train_files):
        print "epoch number " + str(i) + " file number " + str(j)
        X_train, Y_train = get_data_from_file(file)
        y_pred = model.predict(X_test)
        print("auc : " + str(roc_auc_score(Y_test, y_pred)))

        #X_train = normalize(X_train, true_x_test_mean, true_x_test_std)
        #print "X_train shape" + str(X_train.shape)
        #print "Ys labeled as 1s: " + str(Y_train.sum())
        model.fit(X_train, Y_train, class_weight = {0:1.,1:10.}, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])

        model.save(model_path + 'jc_sampleresnet18_v0.hdf5')
        del X_train
        del Y_train