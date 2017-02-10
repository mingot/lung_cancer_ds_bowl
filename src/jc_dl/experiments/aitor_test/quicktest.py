import numpy as np
from keras.optimizers import Adam
from keras import backend as K
import os
from networks.unet import UNETArchitecture

K.set_image_dim_ordering('th')

# PARAMETERS
# smooth: No he llegit que es, poso el que estava al tutorial
# nb_epoch: numero de epocas entrenament
# batch_size: numero de PACIENTS per batch d'entrenament
smooth     = 1.
nb_epoch   = 1
batch_size = 1
input_path = '/home/aiorla/test_input'
output_path = '/home/aiorla/test_output'

# def miniBatchGeneratorLUNA(dataset, index, number_of_patients):
# 	mylist = os.listdir('/mnt/hd2/preprocessed2/')
# 	file_list = [g for g in mylist if g.startswith('luna_')]
# 	file_list.sort()
# 	file_list = file_list[index: index + number_of_patients]
# 	ret = []
# 	for filename in file_list:
# 		b = np.load(os.path.join(mypath,filename))['arr_0']
# 		ret.append(b)
#     return ret

def load_patients(index,num_patients,filelist):
    X_batch, Y_batch = np.empty((num_patients,800,800)), np.empty((num_patients,800,800))
    for filename in file_list:
        b = np.load(os.path.join(input_path,filename))['arr_0']
    return X_batch, Y_batch

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# LOAD "LOCATION" TRAINING DATA
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]

# LOAD MODEL
arch = UNETArchitecture((1,800,800))
model = arch.get_model()

# TRAIN MODEL
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

for e in range(nb_epoch):
    print("epoch %d" % e)
    for i in range(0,len(file_list),batch_size):
        X_batch, Y_batch = load_patients(i,batch_size,file_list)
        model.train(X_batch, Y_batch)

# LOAD & PREDICT & SAVE TEST DATA
