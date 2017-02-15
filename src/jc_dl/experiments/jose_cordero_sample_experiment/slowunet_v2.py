#   SLOWUNET
#   - Go through all the dataset batch by batch
#   - TODO: NOT train slices without "considerable" portion of segmented lung
#   - TODO: Adapt to DL-Pipelines framework (when it is stable...)

import os
import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from networks.unet import UNETArchitecture
from datasets.basic_dataset import LunaNonEmptyMasked_SlicesDataset
from experiments.jose_cordero_sample_experiment.experiments_utils import visualize_case, dice_coef_loss
K.set_image_dim_ordering('th')

# PARAMETERS
num_epoch = 10
max_data_chunk = 2
max_batch_size = 1
prefixes_to_load = ['luna_']
input_paths = ['../../data/sample_data']#/mnt/hd2/preprocessed2']

# PATHS
model_path  = '/home/jose/kaggle/cfis/lung_cancer_ds_bowl/models'

# LOAD LUNA DATASET
dataset = LunaNonEmptyMasked_SlicesDataset(prefixes_to_load, input_paths)
# DEFINE NORMALIZE FUNCTION (PROVISIONAL)
normalize = lambda x: (x - np.mean(x))/np.std(x)

# LOAD MODEL
arch = UNETArchitecture((1,512,512),True)
model = arch.get_model()

## COMPILE THE MODEL
model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy', metrics=['binary_crossentropy', dice_coef_loss])

## TRAIN
for i_epoch in range(num_epoch):
    print("Current epoch " + str(i_epoch))

    ## TRAIN CHUNK BY CHUNK
    for is_valid, (X, Y_mask, Y) in dataset.get_data('train', max_data_chunk, normalize):
        if is_valid:
            model.fit(X,Y_mask,verbose = 0, nb_epoch=1,batch_size=max_batch_size)

    ## SAVE MODEL AFTER 1 EPOCH
    #model.save(model_path+'/slowunet.hdf5')

    ## VALIDATE THE MODEL ON VALID SET
    vals_metrics = []
    for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', max_data_chunk, normalize):
        if is_valid:
            vals_metrics.append(model.evaluate(X,Y_mask,batch_size=max_batch_size))
    print("Validation loss " + str(np.mean(vals_metrics)))


# VISUALIZE RESULTS
for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', max_data_chunk, normalize):
    visualize_case(X,Y_mask,model)
    break
