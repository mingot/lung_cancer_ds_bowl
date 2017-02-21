#   JM_SLOWUNET


import os
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from networks.unet import UNETArchitecture
from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler
from skimage import measure

K.set_image_dim_ordering('th')

# PARAMETERS
NUM_EPOCHS = 10
BATCH_SIZE = 2
USE_EXISTING = False  # load previous model to continue training


## paths
wp = os.environ['LUNG_PATH']
model_path  = wp + 'models/'
#input_path = wp + 'data/preprocessed3_small' #/mnt/hd2/preprocessed2'
input_path = '/mnt/hd2/preprocessed4'
logs_path = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard


## model loading
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    intersection = K.sum(y_true_f * y_pred_f)  # np.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)  # -(2. * intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.0)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

print 'creating model...'
arch = UNETArchitecture((1,512,512),False)
lrate = LearningRateScheduler(step_decay)
model = arch.get_model()
model.compile(optimizer=Adam(lr=1.0e-6), loss=dice_coef_loss, metrics=[dice_coef_loss])

if USE_EXISTING:
    print 'loading model...'
    model.load_weights(model_path + 'jm_slowunet_v4.hdf5')


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
X_test, Y_test = load_patients(file_list[-15:])
file_list = file_list[:-15]


print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    random.shuffle(file_list)
    print 'Epoch: %d/%d' % (i, NUM_EPOCHS)
    for j in range(43):
        X_train, Y_train = load_patients(file_list[j*20:(j+1)*20])
        model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=10, validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])
    model.save(model_path + 'jm_slowunet_v5.hdf5')



def get_regions(nodule_mask):
    thr = np.where(nodule_mask < np.mean(nodule_mask), 0., 1.0)  # threshold detected regions
    label_image = measure.label(thr)  # label them
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    return regions

def intersection_regions(r1, r2):
    h = min(r1.bbox[2], r2.bbox[2]) - max(r1.bbox[0], r2.bbox[0])
    w = min(r1.bbox[3], r2.bbox[3]) - max(r1.bbox[1], r2.bbox[1])
    intersectionArea = w*h
    if h<0 or w<0:
        return 0.0

    area1 = (r1.bbox[2] - r1.bbox[0])*(r1.bbox[3] - r1.bbox[1])
    area2 = (r2.bbox[2] - r2.bbox[0])*(r2.bbox[3] - r2.bbox[1])
    unionArea = area1 + area2 - intersectionArea
    overlapArea = intersectionArea*1.0/unionArea # This should be greater than 0.5 to consider it as a valid detection.
    return overlapArea


#
# tp, fp, fn = 0, 0, 0
# for j in range(20):
#     X_test, Y_test = load_patients(file_list[j*10:(j+1)*10])
#
#     print 'Predicting... %d' % j
#     pred = model.predict([X_test], verbose=0)
#
#     # # plots
#     # idx = 1
#     # plt.imshow(pred[idx,0])
#     # plt.show()
#     # plot_mask(X_test[idx,0], pred[idx,0])
#     # plot_mask(X_test[idx,0], Y_test[idx,0])
#
#     print 'Evaluating... %d' % j
#     for i in range(pred.shape[0]):
#         regions_pred = get_regions(pred[i,0])
#         regions_real = get_regions(Y_test[i,0])
#         for region_real in regions_real:
#             detected = False
#             for region_pred in regions_pred:
#                 # discard regions that occupy everything
#                 if region_real.bbox[0]==0 or region_pred.bbox[0]==0:
#                     continue
#                 score = intersection_regions(r1=region_pred, r2=region_real)
#                 print 'i:%d, score:%s' % (i, str(score))
#                 if score>.5:
#                     tp+=1
#                     detected = True
#                 else:
#                     fp+=1
#             if not detected:
#                 fn += 1
#
#     print 'tp:%d, fp:%d, fn:%d' % (tp,fp,fn)


# VISUALIZE RESULTS
# from experiments.jose_cordero_sample_experiment.experiments_utils import visualize_case
# for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', max_data_chunk, normalize):
#     visualize_case(X,Y_mask,model)
#     break
