#   JM_SLOWUNET

import os
from time import time
import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from networks.unet import UNETArchitecture
# from networks.unet_simplified import UNETArchitecture
from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
# sys.path.remove('/Users/mingot/Projectes/kaggle/ds_bowl_lung/src')
# sys.path.append('/Users/mingot/Projectes/kaggle/ds_bowl_lung/src/jc_dl/')

K.set_image_dim_ordering('th')

# PARAMETERS
NUM_EPOCHS = 30
BATCH_SIZE = 2
TEST_SIZE = 15
USE_EXISTING = False  # load previous model to continue training
OUTPUT_MODEL = 'jm_slowunet_v6_sigmoid.hdf5'


## paths
wp = os.environ['LUNG_PATH']
model_path  = wp + 'models/'
# input_path = wp + 'data/preprocessed3_small' #/mnt/hd2/preprocessed2'
input_path = '/mnt/hd2/preprocessed5'
logs_path = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=True)  # replace keras.callbacks.TensorBoard



# MODEL LOADING -----------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Flatten, Dense, Activation


def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    intersection = K.sum(y_true_f * y_pred_f)  # np.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)  # -(2. * intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.0)


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

print 'creating model...'
#arch = UNETArchitecture((1,512,512),False)
model = get_model(inp_shape=(1,512,512), activation='relu', init='glorot_normal')
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef_loss])
model_checkpoint = ModelCheckpoint(model_path + OUTPUT_MODEL, monitor='loss', save_best_only=True)

if USE_EXISTING:
    print 'loading model...'
    model.load_weights(model_path + 'jm_slowunet_v4.hdf5')



# DATA LOADING AND NORMALIZATION -----------------------------------------------------------------

# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def load_patients(filelist):

    X, Y = [], []
    for filename in filelist:

        b = np.load(os.path.join(input_path, filename))['arr_0']
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
            if lung_volume_l < 0.02 or lung_volume_l > 0.1:
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
            Y.append(nodules_mask)


            if len(slices)>5:  # at most 6 slices per patient
                break
        logging.info('patient %s added %d slices: %s' % (filename, len(slices), str(slices)))
        # if len(X)==0:
        #     continue

        # print len(X), len(Y)

        # p = np.random.permutation(len(X)) # generate permutation of slices
        # X = (np.asarray(X)[p,:,:])[0:2,:,:] #apply permutation
        # Y = (np.asarray(Y)[p,:,:])[0:2,:,:] #apply permutation
    X = np.expand_dims(np.asarray(X),axis=1)
    Y = np.expand_dims(np.asarray(Y),axis=1)
    return X, Y


# Async data loader (this should be improved and go to a separate file)
import multiprocessing
import time
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# Async IO reader class
class IOReader(multiprocessing.Process):
    def __init__(self, max_buffer_size, f):
        super(IOReader, self).__init__()
        self.queue = multiprocessing.Queue(max_buffer_size)
        self.f = f

    def run(self):
        # logging.info('IOReader started')
        for x in self.f():  # TODO modify the constructor to allow pass more arguments to f
           # logging.info('IOReader: readed new data')
           self.queue.put(x)
        self.queue.put('end')
        # logging.info('ending IO reader')
        return

    def get(self):
        # TODO improve this function to check that the process has not died?
        data = self.queue.get()
        if type(data)==str and data=='end':
            raise StopIteration('Finished IO DATA')
        return data


# helper
def data_loader():
    for x in load_patients([file_list[0]]):
        yield x



# TRAINING NETWORK --------------------------------------------------

import random
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]
random.shuffle(file_list)
# file_list = file_list[0:3]

print 'Creating test set...'
X_test, Y_test = load_patients(file_list[-TEST_SIZE:])
print len(X_test), len(Y_test)
file_list = file_list[:-TEST_SIZE]


# print('Training...\n')
# for i in range(NUM_EPOCHS):
#     #random.shuffle(file_list)
#
#     reader = IOReader(2, # number of items to be stored on the buffer
#         data_loader) #function that reads the input and returns a generator
#     reader.start() # start reader on background and fill the buffer
#
#     print 'file list length: %d' % len(file_list)
#     while True:
#         try:
#             logging.info('Epoch: %d/%d' % (i, NUM_EPOCHS))
#             X_train, Y_train = reader.get()
#             logging.info('trainer: Got data from IOReader')
#             logging.info('size: %s %s' % (str(X_train.shape), str(Y_train.shape)))
#             model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE, shuffle=True) #, callbacks=[tb])
#         except StopIteration:
#             break
#
#     model.save(model_path + OUTPUT_MODEL)
#     logging.info("Ending processing one epoch" )
#     reader.join()


print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    random.shuffle(file_list)
    for j in range(43):
        logging.info('Epoch: %d/%d, batch:%d' % (i, NUM_EPOCHS, j*20))
        X_train, Y_train = load_patients(file_list[j*20:(j+1)*20])
        print X_train.shape, Y_train.shape
        model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE,
                  validation_data=(X_test, Y_test), shuffle=True, callbacks=[tb])
    model.save(model_path + OUTPUT_MODEL)


# TESTING NETWORK --------------------------------------------------

# def get_regions(nodule_mask):
#     thr = np.where(nodule_mask < np.mean(nodule_mask), 0., 1.0)  # threshold detected regions
#     label_image = measure.label(thr)  # label them
#     labels = label_image.astype(int)
#     regions = measure.regionprops(labels)
#     return regions
#
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
#

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
