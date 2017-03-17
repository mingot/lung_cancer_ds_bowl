import os
import csv
import sys
import copy
import math
import random
import os.path
import numpy as np
from time import time
from keras import backend as K
from keras.optimizers import Adam
from numpy.random import binomial
from sklearn.metrics import roc_auc_score
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/")
sys.path.append("/home/felix/lung_cancer_ds_bowl/src/jc_dl/")
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard

input_path = '/mnt/hd2/preprocessed4'
model_path = '/mnt/hd2/models/'
BATCH_SIZE = 20

model = ResnetBuilder().build_resnet_18((512,1,512),1)
model.load_weights(model_path + 'jc_sampleresnet18_v0.hdf5')

fieldnames = ['id', 'min','mean','max']
file_list = os.listdir(input_path)
len_list = len(file_list)

with open('output.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, filename in enumerate(file_list):
        print("processing file number " +str(i)+" of "+str(len_list))
        X = np.load(os.path.join(input_path, filename))['arr_0'][0,:,:,:]
        Y = model.predict(X,batch_size=BATCH_SIZE)
        min_, max_, mean_, std_ = K.min(Y), K.max(Y), K.mean(Y), K.std(Y)
        writer.writerow({'id': filename, 'min': min_, 'max': max_, 'mean':mean_, 'std':std_})
        del X
        del Y
