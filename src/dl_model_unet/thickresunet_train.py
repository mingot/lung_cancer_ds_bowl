import os
import random
import sys
sys.path.append('/home/mteixido/lung_cancer_ds_bowl/src/dl_networks')
sys.path.append('/home/mteixido/lung_cancer_ds_bowl/src')

import os
import random
from thickresunet import ThickRESUNET

# PARAMETERS -----------------------------------
input_path = '/mnt/hd2/preprocessed5'
logs_path = '/home/mteixido/logs'
TEST_SIZE = 3 # in number of patients
NUM_EPOCHS = 10
SAMPLES_PER_EPOCH = 300 # number of thick slices to train per epoch
BATCH_SIZE = 6  # batch size should not be too long to avoid memory crash on GPU


# DATASET CREATION --------------------------------
file_list = [g for g in os.listdir(input_path) if g.startswith('luna_')]
file_list = [os.path.join(input_path, f) for f in file_list] # better save the absolute name
random.shuffle(file_list)

test_patients = file_list[-TEST_SIZE:] # select some patients for validation
train_patients = file_list[:-TEST_SIZE] #all files except validation ones

print test_patients

model = ThickRESUNET(input_shape=(3,512,512), lr=1.0e-3) # load a file to continue a training (optional)

# PARAMETERS -----------------------------------
input_path = '/mnt/hd2/preprocessed5'
logs_path = '/home/mteixido/logs'
modeloutput_file = '/home/mteixido/thickresmodel3.hdf5'
TEST_SIZE = 3 # in number of patients
NUM_EPOCHS = 20
SAMPLES_PER_EPOCH = 300 # number of thick slices to train per epoch
BATCH_SIZE = 5  # batch size should not be too long to avoid memory crash on GPU


model.train(nb_epochs=NUM_EPOCHS,  #number of epochs to train
              samples_per_epoch=SAMPLES_PER_EPOCH, #epoch is defined as a fixed number of samples
              batch_size=BATCH_SIZE, # batch size
              train_file_list=train_patients, # list of training filenames
              validation_file_list=test_patients, #validation data
              logs_path=logs_path,
              modeloutput_path=modeloutput_file, # where to save the file
              max_q_size=10,  #length of queue in number of batches to be ready
              nb_worker=1,    #number of parallel threads
              max_slices=4)   # to increase suffling
