#   SLOWUNET
#   - Go through all the dataset batch by batch
#   - TODO: NOT train slices without "considerable" portion of segmented lung
#   - TODO: Adapt to DL-Pipelines framework (when it is stable...)

import os
import numpy as np
import keras
from time import time
from keras.optimizers import Adam
from keras import backend as K
from networks.unet import UNETArchitecture
from datasets.basic_dataset import LunaNonEmptyMasked_SlicesDataset
from experiments.jose_cordero_sample_experiment.experiments_utils import visualize_case #, dice_coef_loss
K.set_image_dim_ordering('th')

wp = os.environ['LUNG_PATH']
print wp

logdir = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(logdir):
    os.makedirs(logdir)
tb = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=10, write_graph=True, write_images=True)

# PARAMETERS
num_epoch = 10
max_data_chunk = 2
max_batch_size = 2
prefixes_to_load = ['luna_']
input_paths = ['../../data/preprocessed3_small']#/mnt/hd2/preprocessed2']

# PATHS
model_path  = '/home/jose/kaggle/cfis/lung_cancer_ds_bowl/models'

# LOAD LUNA DATASET
dataset = LunaNonEmptyMasked_SlicesDataset(prefixes_to_load, input_paths)
# DEFINE NORMALIZE FUNCTION (PROVISIONAL)
normalize = lambda x: (x - np.mean(x))/np.std(x)

# LOAD MODEL
arch = UNETArchitecture((1,512,512),False)
model = arch.get_model()



def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = 1000*K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = 1000*np.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.0)


## COMPILE THE MODEL
# model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy', metrics=['binary_crossentropy', dice_coef_loss])
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef_loss])


## TRAIN
for i_epoch in range(num_epoch):
    print("Current epoch " + str(i_epoch))

    ## TRAIN CHUNK BY CHUNK
    for is_valid, (X, Y_mask, Y) in dataset.get_data('train', 2, normalize):
        if is_valid:
            #print X.shape
            #print Y_mask.shape
            model.fit(X, Y_mask, verbose=1, nb_epoch=1, batch_size=2, shuffle=True, callbacks=[tb])
            #break
        # if i%%10==0:
        #     print 'Predicting'
        #     Y_pred = model.predict(X)
        #     plt.imshow(X[0,0])
        #     plt.show()
        #     plt.imshow(Y_mask[0,0])
        #     plt.show()
        #     plotting.plot_mask(X[0,0], Y_mask[0,0])
        #     plt.imshow(Y_pred[0,0])
        #     plt.show()
        #     dice_coef_np(y_true=Y_mask[0,0].astype(np.float32), y_pred=Y_pred[0,0])
        #     #Y_mask.astype(np.float32)
        #     np.save()

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
