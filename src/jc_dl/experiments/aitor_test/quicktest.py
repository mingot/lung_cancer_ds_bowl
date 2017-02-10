import numpy as np
from keras.optimizers import Adam
from keras import backend as K
import os
from networks.unet import UNETArchitecture

K.set_image_dim_ordering('th')
smooth     = 1.

# PARAMETERS
# nb_epoch: # training epochs
# batch_size: # PATIENTS in a batch
nb_epoch   = 1
batch_size = 1
input_path = '/home/aiorla/test_input'
output_path = '/home/aiorla/test_output'
model_path = '/home/aiorla'

def load_patients(index,num_patients,filelist):
    X_batch, Y_batch = [], []
    for i in range(num_patients):
       b = np.load(os.path.join(input_path,filelist[index+i]))['arr_0']
       for j in range(b.shape[1]):
           X_batch.append(b[0,j,:,:])
           Y_batch.append(b[2,j,:,:])
       print(len(X_batch))
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

# TRAIN & SAVE MODEL
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

for e in range(nb_epoch):
    print("epoch %d" % e)
    for i in range(0,len(file_list),batch_size):
        X_batch, Y_batch = load_patients(i,batch_size,file_list)
        #model.train(X_batch, Y_batch) # TOTRY
    model.save(model_path+'/unet-'+str(e)+'.h5')

# LOAD & PREDICT & SAVE TEST DATA
# TODO...
