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
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/dl_model_slices/")
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard
#from dl_model_slices.dataset import normalize

def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    # hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

input_path = '/mnt/hd2/preprocessed5'
model_path = '/mnt/hd2/models/'
BATCH_SIZE = 100

model = ResnetBuilder().build_resnet_50((512,1,512),1)
model.load_weights(model_path + 'jc_sampleresnet18_v1.hdf5')

fieldnames = ['id', 'min','mean','max','std']
file_list = os.listdir(input_path)
len_list = len(file_list)

with open('output.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, filename in enumerate(file_list):
        print("processing file number " +str(i)+" of "+str(len_list))
        X = np.load(os.path.join(input_path, filename))['arr_0'][0,:,:,:]
        X = X.reshape([ X.shape[0], 1, X.shape[1], X.shape[2] ])
	X = normalize(np.array(X))
        Y = model.predict(X,batch_size=BATCH_SIZE)
        min_, max_, mean_, std_ = Y.min(), Y.max(), Y.mean(), Y.std()
        writer.writerow({'id': filename, 'min': min_, 'max': max_, 'mean':mean_, 'std':std_})
        del X
        del Y
