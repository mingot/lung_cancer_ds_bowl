#   JM_SLOWUNET

import os
import random
import numpy as np
import logging
from time import time
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Flatten, Dense, Activation
from dl_utils.tb_callback import TensorBoard
from dl_utils.heatmap import extract_regions_from_heatmap
# sys.path.remove('/Users/mingot/Projectes/kaggle/ds_bowl_lung/src')
# sys.path.append('/Users/mingot/Projectes/kaggle/ds_bowl_lung/src/jc_dl/')
# export PYTHONPATH=/Users/mingot/Projectes/kaggle/ds_bowl_lung/src/jc_dl/


# PARAMETERS
NUM_EPOCHS = 30
BATCH_SIZE = 2
TEST_SIZE = 15  # number of patients to test on
TRAIN = True  # should the model be trained
USE_EXISTING = False  # load previous model to continue training or test


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = wp + 'data/preprocessed5_sample' # '/mnt/hd2/preprocessed5'  #
OUTPUT_MODEL = wp + 'models/AUX_teixi_slowunet_weightloss.hdf5'
OUTPUT_CSV = wp + 'output/AUX_noduls_unet_v03.csv'
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


print 'Creating model...'
model = get_model(inp_shape=(1,512,512), activation='relu', init='glorot_normal')
model.compile(optimizer=Adam(lr=1.0e-5), loss=weighted_loss, metrics=[weighted_loss])

if USE_EXISTING:
    print 'Loading exiting model...'
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

    if shuffle and len(X)>0:
        aux = list(zip(X, Y))
        random.shuffle(aux)
        X, Y = zip(*aux)

    # add extra dimension (necessary for the batch index, internal of keras fit function)
    X = np.expand_dims(np.asarray(X),axis=1)
    Y = np.expand_dims(np.asarray(Y),axis=1)

    return X, Y



def chunks(file_list=[], batch_size=5, infinite=True, full=False):

    CONCURRENT_PATIENTS = 2  # limitation by memory
    while True:
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

# TRAINING NETWORK --------------------------------------------------


if TRAIN:
    file_list = os.listdir(INPUT_PATH)
    file_list = [g for g in file_list if g.startswith('luna_')]
    random.shuffle(file_list)
    #file_list = file_list[0:10]

    print 'Creating test set...'
    file_list_test = file_list[-TEST_SIZE:]
    file_list_train = file_list[:-TEST_SIZE]

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


    model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
    model.fit_generator(generator=chunks(file_list_train,BATCH_SIZE,infinite=True),
                        samples_per_epoch=NUM_EPOCHS, # make it small to update TB and CHECKPOINT frequently
                        nb_epoch=888*3,  # put here the number of samples
                        verbose=1,
                        callbacks=[tb, model_checkpoint],
                        validation_data=chunks(file_list_test,BATCH_SIZE,infinite=True),
                        nb_val_samples=20,  # TO REVIEW
                        max_q_size=10,
                        nb_worker=1)  # a locker is needed if increased the number of parallel workers

# TESTING NETWORK --------------------------------------------------

# Load testing file_list
file_list = os.listdir(INPUT_PATH)
# file_list = [g for g in file_list if g.startswith('luna_')]


print 'Testing...'
with open(OUTPUT_CSV, 'w') as file:

    # write the header
    file.write('filename,nslice,x,y,diameter,max_intensity,min_intensity,mean_intensity\n')

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

            print 'predicting heatmap...'
            heatmap = model.predict([X], verbose=0)
            heatmap = heatmap[0,0]
            regions_pred = extract_regions_from_heatmap(heatmap, threshold=0.8*np.max(heatmap))
            for r in regions_pred:
                file.write('%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,
                                                           r.max_intensity, r.min_intensity, r.mean_intensity))
        print time()-tstart


