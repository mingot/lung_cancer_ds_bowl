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
from utils.tb_callback import TensorBoard

K.set_image_dim_ordering('th')

# PARAMETERS
num_epoch = 10
max_data_chunk = 2
max_batch_size = 2
prefixes_to_load = ['luna_']


## paths
wp = os.environ['LUNG_PATH']
model_path  = wp + 'models/'
#input_path = wp + 'data/preprocessed3_small' #/mnt/hd2/preprocessed2'
input_path = '/mnt/hd2/preprocessed3'
logs_path = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


## tensorboard logs
#tb = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=False, write_images=True)
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)


## model loading
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    intersection = K.sum(y_true_f * y_pred_f)  # np.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)  # -(2. * intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.0)

arch = UNETArchitecture((1,512,512),False)
model = arch.get_model()
model.compile(optimizer=Adam(lr=1.0e-6), loss=dice_coef_loss, metrics=[dice_coef_loss])


## Load LUNA dataset
normalize = lambda x: (x - np.mean(x))/np.std(x)

def load_patients(filelist):
    X, Y = [], []
    for filename in filelist:
        b = np.load(os.path.join(input_path, filename))['arr_0']
        if b.shape[0]!=3:
            continue

        tot = 0
        last_slice = -1e3  # big initialization
        slices = []
        for j in range(b.shape[1]):

            lung_image = b[0,j,:,:]
            lung_mask = b[1,j,:,:]
            nodules_mask = b[2,j,:,:]

            # Discard if no nodules
            if nodules_mask.sum() == 0:
                continue

            # Discard if bad segmentation
            voxel_volume_l = 2*0.7*0.7/(1000000.0)
            lung_volume_l = np.sum(lung_mask)*voxel_volume_l
            if lung_volume_l < 0.02 or lung_volume_l > 0.1:
                continue  # skip slices with bad lung segmentation

            # discard if consecutive slices
            if j<last_slice + 5:
                continue

            # if ok append
            last_slice = j
            slices.append(j)
            tot+=1
            lung_image[lung_mask==0]=-1000  # apply mask
            X.append(normalize(lung_image))
            Y.append(nodules_mask)
            if tot>2:  # at most 3 slices per patient
                break
        print 'patient %s added %d slices: %s' % (filename, tot, str(slices))

    X = np.expand_dims(np.asarray(X),axis=1)
    Y = np.expand_dims(np.asarray(Y),axis=1)
    return X, Y


import random
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]
random.shuffle(file_list)
print 'Creating test set...'
X_test, Y_test = load_patients(file_list[-10:])


NUM_EPOCHS = 10
print('Training...\n')
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    random.shuffle(file_list)
    print 'Epoch: %d/%d' % (i, NUM_EPOCHS)
    model.save(model_path + 'jm_slowunet_v3.hdf5')
    for j in range(30):
        X_train, Y_train = load_patients(file_list[j*20:(j+1)*20])
        model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=2, validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])



# print('Creating validation set...\n')
# X_val = []
# Y_val = []
# i=0
# for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', 1, normalize):
#     if is_valid:
#         print("Iteration: %d" % i)
#         i+=1
#         X_val.append(X[0,0])
#         Y_val.append(Y_mask[0,0])
# X_val = np.expand_dims(np.asarray(X_val),axis=1)
# Y_val = np.expand_dims(np.asarray(Y_val),axis=1)
#
#
# print('Creating training set...\n')
# X_tot = []
# Y_tot = []
# i=0
# for is_valid, (X, Y_mask, Y) in dataset.get_data('train', 1, normalize):
#     if len(X_tot)>10:
#         break
#     if is_valid:
#         print("Iteration: %d" % i)
#         i+=1
#         X_tot.append(X[0,0])
#         Y_tot.append(Y_mask[0,0])
# X_tot = np.expand_dims(np.asarray(X_tot),axis=1)
# Y_tot = np.expand_dims(np.asarray(Y_tot),axis=1)





# ## TRAIN
# for i_epoch in range(num_epoch):
#     print("Current epoch " + str(i_epoch))
#
#     ## TRAIN CHUNK BY CHUNK
#
#     for is_valid, (X, Y_mask, Y) in dataset.get_data('train', 50, normalize):
#         if is_valid:
#             model.fit(X, Y_mask, verbose=1, nb_epoch=1, batch_size=2, shuffle=True, callbacks=[tb])  # validation_split=.25
#
#         # if i%%10==0:
#         #     print 'Predicting'
#         #     Y_pred = model.predict(X)
#         #     plt.imshow(X[0,0])
#         #     plt.show()
#         #     plt.imshow(Y_mask[0,0])
#         #     plt.show()
#         #     plotting.plot_mask(X[0,0], Y_mask[0,0])
#         #     plt.imshow(Y_pred[0,0])
#         #     plt.show()
#         #     dice_coef_np(y_true=Y_mask[0,0].astype(np.float32), y_pred=Y_pred[0,0])
#         #     #Y_mask.astype(np.float32)
#         #     np.save()
#
#     ## SAVE MODEL AFTER 1 EPOCH
#     model.save(model_path + 'jm_slowunet.hdf5')
#
#     ## VALIDATE THE MODEL ON VALID SET
#     vals_metrics = []
#     for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', max_data_chunk, normalize):
#         if is_valid:
#             vals_metrics.append(model.evaluate(X,Y_mask,batch_size=max_batch_size))
#     print("Validation loss " + str(np.mean(vals_metrics)))


# VISUALIZE RESULTS
# for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', max_data_chunk, normalize):
#     visualize_case(X,Y_mask,model)
#     break
