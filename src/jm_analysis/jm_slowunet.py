#   JM_SLOWUNET

import os
from time import time
import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import multiprocessing
import logging
# sys.path.remove('/Users/mingot/Projectes/kaggle/ds_bowl_lung/src')
# sys.path.append('/Users/mingot/Projectes/kaggle/ds_bowl_lung/src/jc_dl/')
# export PYTHONPATH=/Users/mingot/Projectes/kaggle/ds_bowl_lung/src/jc_dl/


# PARAMETERS
NUM_EPOCHS = 30
BATCH_SIZE = 2
TEST_SIZE = 15
USE_EXISTING = True  # load previous model to continue training


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # wp + 'data/preprocessed5_sample'
OUTPUT_MODEL = wp + 'models/teixi_slowunet_weightloss.hdf5'
OUTPUT_CSV = wp + 'output/nodules_unet/noduls_unet_v02.csv'
LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# MODEL LOADING -----------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Flatten, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU

def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    intersection = K.sum(y_true_f * y_pred_f)  # np.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)  # -(2. * intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.0)

def weighted_loss(y_true, y_pred, pos_weight=100):
    #if this argument is greater than 1 we will penalize more the nodules not detected as nodules, we can set it up to 10 or 100?
     y_true_f = K.flatten(y_true)  # y_true.flatten()
     y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
     return K.mean(-(1-y_true_f)*K.log(1-y_pred_f)-y_true_f*K.log(y_pred_f)*pos_weight)

def get_model(inp_shape, activation='relu', init='glorot_normal'):
    inputs = Input(inp_shape)

    conv1 = Convolution2D(32, 3, 3, activation=activation, init=init, border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation=activation, init=init, border_mode='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation=activation, init=init, border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation=activation, init=init, border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation=activation, init=init, border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation=activation, init=init, border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation=activation, init=init, border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation=activation, init=init, border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation=activation, init=init, border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation=activation, init=init, border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation=activation, init=init, border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation=activation, init=init, border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation=activation, init=init, border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation=activation, init=init, border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation=activation, init=init, border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation=activation, init=init, border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation=activation, init=init, border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation=activation, init=init, border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    # TEMPORAL: test average pooling as suggested by OriolV
    # avpooling = AveragePooling2D(pool_size=(512, 512))(conv10)
    # aa = Flatten(name="flatten")(avpooling)
    # bb = Dense(1, activation='sigmoid', name='dense_3')(aa)
    #cc = Activation("softmax",name="softmax")(bb)

    return Model(input=inputs, output=conv10)  # conv10

def get_model_soft(inp_shape):
    inputs = Input(inp_shape)
    init = 'glorot_normal'

    conv1 = Convolution2D(32, 3, 3, init=init, border_mode='same')(inputs)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Convolution2D(32, 3, 3, init=init, border_mode='same')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, init=init, border_mode='same')(pool1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Convolution2D(64, 3, 3, init=init, border_mode='same')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, init=init, border_mode='same')(pool2)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Convolution2D(128, 3, 3, init=init, border_mode='same')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, init=init, border_mode='same')(pool3)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Convolution2D(256, 3, 3, init=init, border_mode='same')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, init=init, border_mode='same')(pool4)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Convolution2D(512, 3, 3, init=init, border_mode='same')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, init=init, border_mode='same')(up6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
    conv6 = Convolution2D(256, 3, 3, init=init, border_mode='same')(conv6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, init=init, border_mode='same')(up7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)
    conv7 = Convolution2D(128, 3, 3, init=init, border_mode='same')(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, init=init, border_mode='same')(up8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)
    conv8 = Convolution2D(64, 3, 3, init=init, border_mode='same')(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, init=init, border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, init=init, border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    return Model(input=inputs, output=conv10)

print 'creating model...'
#arch = UNETArchitecture((1,512,512),False)
model = get_model(inp_shape=(1,512,512), activation='relu', init='glorot_normal')
#model = get_model_soft(inp_shape=(1,512,512))
model.compile(optimizer=Adam(lr=1.0e-5), loss=weighted_loss, metrics=[weighted_loss])
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)

if USE_EXISTING:
    print 'loading model...'
    model.load_weights(OUTPUT_MODEL)



# DATA LOADING AND NORMALIZATION -----------------------------------------------------------------

# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def load_patients(filelist, full=False, shuffle=False):

    X, Y = [], []
    for filename in filelist:

        b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
        if b.shape[0]!=3:
            continue

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
            if lung_volume_l < 0.02 or lung_volume_l > 0.1 and full is not False:
                continue  # skip slices with bad lung segmentation

            # discard if consecutive slices
            if j<last_slice + 5:
                continue

            # # discard if nodules out of lungs
            # if np.any(np.logical_and(nodules_mask, 0 == lung_mask)):
            #     print 'nodules out of lungs for %s at %d' % (filename, j)
            #     continue

            # if ok append
            last_slice = j
            slices.append(j)
            lung_image[lung_mask==0]=-1000  # apply mask
            X.append(normalize(lung_image))
            Y.append(nodules_mask)  # nodules_mask


            if len(slices)>5:  # at most 6 slices per patient
                break
        logging.info('patient %s added %d slices: %s' % (filename, len(slices), str(slices)))
        # if len(X)==0:
        #     continue

        # print len(X), len(Y)


    X = np.expand_dims(np.asarray(X),axis=1)
    Y = np.expand_dims(np.asarray(Y),axis=1)

    if shuffle:  # TO review!!
        p = np.random.permutation(len(X)) # generate permutation of slices
        X = (np.asarray(X)[p,:,:])[0:2,:,:] #apply permutation
        Y = (np.asarray(Y)[p,:,:])[0:2,:,:] #apply permutation

    return X, Y


##

def chunks(file_list=[], batch_size=5, infinite=True, full=False):

    CONCURRENT_PATIENTS = 2  # limitation by memory
    while True:
        # for filename in file_list:
        for j in range(0, len(file_list), CONCURRENT_PATIENTS):
            filenames = file_list[j:j+CONCURRENT_PATIENTS]
            a, b = load_patients(filenames,full=False,shuffle=True)
            if len(a.shape) != 4: # xapussa per evitar casos raros com luna_243094273518213382155770295147.npz que li passa a aquest???
                continue

            size = a.shape[0]
            num_batches = int(np.ceil(size / float(batch_size)))
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size  # min(size, (i + 1) * batch_size)
                yield a[start:end], b[start:end]

        if infinite is False:
            break

# # TRAINING NETWORK --------------------------------------------------
#
# import random
# mylist = os.listdir(INPUT_PATH)
# file_list = [g for g in mylist if g.startswith('luna_')]
# random.shuffle(file_list)
# #file_list = file_list[0:10]
#
# print 'Creating test set...'
# X_test, Y_test = load_patients(file_list[-TEST_SIZE:])
# file_list = file_list[:-TEST_SIZE]
#
#
# # print('Training...\n')
# # for i in range(NUM_EPOCHS):
# #     #random.shuffle(file_list)
# #
# #     reader = IOReader(2, # number of items to be stored on the buffer
# #         data_loader) #function that reads the input and returns a generator
# #     reader.start() # start reader on background and fill the buffer
# #
# #     print 'file list length: %d' % len(file_list)
# #     while True:
# #         try:
# #             logging.info('Epoch: %d/%d' % (i, NUM_EPOCHS))
# #             X_train, Y_train = reader.get()
# #             logging.info('trainer: Got data from IOReader')
# #             logging.info('size: %s %s' % (str(X_train.shape), str(Y_train.shape)))
# #             model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE, shuffle=True) #, callbacks=[tb])
# #         except StopIteration:
# #             break
# #
# #     model.save(model_path + OUTPUT_MODEL)
# #     logging.info("Ending processing one epoch" )
# #     reader.join()
#
#
# # print 'Training...'
# # for i in range(NUM_EPOCHS):
# #     random.shuffle(file_list)
# #     X_train, Y_train = load_patients(file_list)
# #     model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE,
# #                    shuffle=True, callbacks=[tb])  # validation_data=(X_test, Y_test),
#
#
# print('Training...\n')
# # model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
# for i in range(NUM_EPOCHS):
#     random.shuffle(file_list)
#     for j in range(43):
#         logging.info('Epoch: %d/%d, batch:%d' % (i, NUM_EPOCHS, j*20))
#         X_train, Y_train = load_patients(file_list[j*20:(j+1)*20])
#         model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE,
#                   validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])
#     model.save(model_path + OUTPUT_MODEL)


model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
model.fit_generator(generator=chunks(file_list_train,BATCH_SIZE,infinite=True),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=???,
    verbose=1,
    callbacks=[tb, model_checkpoint],
    validation_data=chunks(file_list_test,BATCH_SIZE,infinite=True),
    nb_val_samples=10,  # TO REVIEW
    max_q_size=10,
    nb_worker=2)

# TESTING NETWORK --------------------------------------------------
from skimage import measure


def get_regions(nodule_mask):
    # thr = np.where(nodule_mask < np.mean(nodule_mask), 0., 1.0)  # threshold detected regions
    thr = np.where(nodule_mask < 0.6*np.max(nodule_mask), 0., 1.0)  # threshold detected regions
    label_image = measure.label(thr)  # label them
    labels = label_image.astype(int)
    regions = measure.regionprops(labels, nodule_mask)
    return regions

# def intersection_regions(r1, r2):
#     h = min(r1.bbox[2], r2.bbox[2]) - max(r1.bbox[0], r2.bbox[0])
#     w = min(r1.bbox[3], r2.bbox[3]) - max(r1.bbox[1], r2.bbox[1])
#     intersectionArea = w*h
#     if h<0 or w<0:
#         return 0.0
#
#     area1 = (r1.bbox[2] - r1.bbox[0])*(r1.bbox[3] - r1.bbox[1])
#     area2 = (r2.bbox[2] - r2.bbox[0])*(r2.bbox[3] - r2.bbox[1])
#     unionArea = area1 + area2 - intersectionArea
#     overlapArea = intersectionArea*1.0/unionArea # This should be greater than 0.5 to consider it as a valid detection.
#     return overlapArea


from time import time
file_list = os.listdir(INPUT_PATH)
# file_list = [g for g in file_list if g.startswith('luna_')]


with open(OUTPUT_CSV) as file:


    for idx, filename in enumerate(file_list):
        tstart = time()


        b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
        X = []
        print 'Patient: %s (%d/%d)' % (filename, idx, len(file_list))
        for nslice in range(b.shape[1]):
            if nslice%3 in [0,1]:
                continue
            X = []
            lung_image = b[0,nslice,:,:]
            lung_mask = b[1,nslice,:,:]
            lung_image[lung_mask==0]=-1000  # apply mask
            X.append(normalize(lung_image))
            X = np.expand_dims(np.asarray(X),axis=1)

            pred = model.predict([X], verbose=0)

            regions_pred = get_regions(pred[0,0])
            for r in regions_pred:
                # print '%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,
                #                                            r.max_intensity, r.min_intensity, r.mean_intensity)

                file.write('%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,
                                                           r.max_intensity, r.min_intensity, r.mean_intensity))
        print time()-tstart




# tp, fp, fn = 0, 0, 0
# for j in range(20):
#     X_test, Y_test = load_patients(file_list[j*10:(j+1)*10])
#
#     print 'Predicting... %d' % j
#     pred = model.predict([X_test], verbose=0)
#
#     # # plots
#     # x = pred[3,0]*10000
#     # x = (x-np.min(x))/(np.max(x)-np.min(x))
#     # plt.imshow(x)
#     # plt.show()
#     #
#     # idx = 4
#     # plt.imshow(pred[idx,0])
#     # plt.show()
#     # plot_mask(X_test[idx,0], pred[idx,0])
#     # plot_mask(X_test[idx,0], Y_test[idx,0])
#
#     print 'Evaluating... %d' % j
#     for i in range(pred.shape[0]):
#         regions_pred = get_regions(pred[i,0])
#         regions_real = get_regions(Y_test[i,0])
#
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
