#   QUICKUNET
#   - Tutorial-like UNET training, using only a subset of LUNA patients (controlled with num_patient)
#   - TODO: NOT train slices without "considerable" portion of segmented lung
#   - TODO: Adapt to DL-Pipelines framework (when it is stable...)

import os
import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from networks.unet import UNETArchitecture

K.set_image_dim_ordering('th')

# PARAMETERS
num_epoch   = 1
num_patient = 2 # RAM  limited
batch_size  = 1 # VRAM limited

# PATHS
input_path  = '/home/aiorla/Projects/Lung/test/input'
model_path  = '/home/aiorla/Projects/Lung/test/model'
output_path = '/home/aiorla/Projects/Lung/test/output'

# I/O METHOD
def load_patients(index,num_patients,filelist):
    X, Y = [], []
    for i in range(num_patients):
       b = np.load(os.path.join(input_path,filelist[min(index+i,len(filelist)-1)]))['arr_0']
       for j in range(b.shape[1]):
           X.append(b[0,j,:,:])
           Y.append(b[2,j,:,:])
    X = np.expand_dims(np.asarray(X),axis=1)
    Y = np.expand_dims(np.asarray(Y),axis=1)
    return X, Y

# MODEL METHODS
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# LOAD DATA LOCATION
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]

# LOAD MODEL
arch = UNETArchitecture((1,512,512),False)
model = arch.get_model()

# TRAIN
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef_loss])
for e in range(num_epoch):
    print "epoch " + str(e)
    X, Y = load_patients(0,num_patient,file_list)
    model.fit(X,Y,nb_epoch=1,batch_size=batch_size)
    model.save(model_path+'/quickunet.hdf5')
