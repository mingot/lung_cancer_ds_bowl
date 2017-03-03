#   Modificacions a JM_SLOWNET
import os
from time import time
import numpy as np
from keras.optimizers import Adam
from keras import backend as K
#from networks.unet import UNETArchitecture
# from networks.unet_simplified import UNETArchitecture
from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

K.set_image_dim_ordering('th')

# PARAMETERS
NUM_EPOCHS = 30
BATCH_SIZE = 2
TEST_SIZE = 15
USE_EXISTING = True  # load previous model to continue training
OUTPUT_MODEL = 'teixi_thickunet_weightloss.hdf5'


## paths
wp = '/home/mteixido/'
model_path  = wp + 'models/'
input_path = '/mnt/hd2/preprocessed5'
logs_path = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard



# MODEL LOADING -----------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Flatten, Dense, Activation


def weighted_loss(y_true, y_pred, pos_weight=100):
    #if this argument is greater than 1 we will penalize more the nodules not detected as nodules, we can set it up to 10 or 100?
     y_true_f = K.flatten(y_true)  # y_true.flatten()
     y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
     return K.mean(-(1-y_true_f)*K.log(1-y_pred_f)-y_true_f*K.log(y_pred_f)*pos_weight)


from keras.layers.advanced_activations import LeakyReLU

def get_model(inp_shape, activation='relu', init='glorot_normal', first_depth=32):
    inputs = Input(inp_shape)

    conv1 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(inputs)
    conv1 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(pool1)
    conv2 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(pool2)
    conv3 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(pool3)
    conv4 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(first_depth*16, 3, 3, activation=activation, init=init, border_mode='same')(pool4)
    conv5 = Convolution2D(first_depth*16, 3, 3, activation=activation, init=init, border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(up6)
    conv6 = Convolution2D(first_depth*8, 3, 3, activation=activation, init=init, border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(up7)
    conv7 = Convolution2D(first_depth*4, 3, 3, activation=activation, init=init, border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(up8)
    conv8 = Convolution2D(first_depth*2, 3, 3, activation=activation, init=init, border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(up9)
    conv9 = Convolution2D(first_depth, 3, 3, activation=activation, init=init, border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    return Model(input=inputs, output=conv10)  # conv10

print 'creating model...'
#arch = UNETArchitecture((1,512,512),False)
model = get_model(inp_shape=(5,512,512), activation='relu', init='glorot_normal', first_depth=8)
model.compile(optimizer=Adam(lr=1.0e-5), loss=weighted_loss, metrics=[weighted_loss])
model_checkpoint = ModelCheckpoint(model_path + OUTPUT_MODEL, monitor='loss', save_best_only=True)

# DATA LOADING AND NORMALIZATION -----------------------------------------------------------------

# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def load_patients(filelist,thickness=5):
    X, Y = [], []
    for filename in filelist:
        b = np.load(os.path.join(input_path, filename))['arr_0']
        if b.shape[0]!=3: # si falta la nodule_mask asumim 0's
            x = b.shape[1:]
            b=np.concatenate(([b[0,:,:,:]],[b[1,:,:,:]],[np.zeros(x)]),axis=0)

        last_slice = -1e3  # big initialization
        slices = []
        max_slice=b.shape[1]
        d = (thickness-1)/2

        for j in range(d,max_slice-d,thickness):

            lung_image = b[0,(j-d):(j+d+1),:,:]
            lung_mask = b[1,(j-d):(j+d+1),:,:]
            nodules_mask = b[2,(j-d):(j+d+1),:,:]

            # Discard if no nodules
            #if nodules_mask.sum() == 0:
            #    continue

            # Discard if bad segmentation
            voxel_volume_l = 2*0.7*0.7/(1000000.0)
            lung_volume_l = np.sum(lung_mask)/thickness*voxel_volume_l
            if lung_volume_l < 0.02 or lung_volume_l > 0.1:
                continue  # skip slices with bad lung segmentation

            # discard if consecutive slices
            #if j<last_slice + 5:
            #    continue

            # if ok append
            last_slice = j
            slices.append(j)
            lung_image[lung_mask==0]=-1000  # apply mask
            X.append(normalize(lung_image))
            Y.append(np.max(nodules_mask,axis=0))  # nodules_mask
            #if len(slices)>5:  # at most 6 slices per patient
            #    break
        logging.info('patient %s added %d slices: %s' % (filename, len(slices), str(slices)))

    #X = np.expand_dims(np.asarray(X),axis=1)
    X = np.asarray(X)
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
    def __init__(self, max_buffer_size, f,*args,**kwargs):
        super(IOReader, self).__init__()
        self.queue = multiprocessing.Queue(max_buffer_size)
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def run(self):
        logging.info('IOReader started')
        for x in self.f(*self.args,**self.kwargs):  
           logging.info('IOReader: readed new data')
           self.queue.put(x)
        self.queue.put('end')
        logging.info('ending IO reader')
        return

    def get(self):
        # TODO improve this function to check that the process has not died?
        data = self.queue.get()
        if type(data)==str and data=='end':
            raise StopIteration('Finished IO DATA')
        return data

    def data_generator(self,max_iters=1e5):
        i=1
        while True:
            try:
                yield self.get()
                i = i+1
                if i > max_iters:
                    self.terminate()
                    raise StopIteration()
            except StopIteration:
                break


# helper
def data_loader(file_list=[]):
    for file in file_list:
        X,Y = load_patients([file])
        if len(X.shape)==4:  # xapussa per evitar casos raros com luna_243094273518213382155770295147.npz que li passa a aquest???
            yield X,Y




# TRAINING NETWORK --------------------------------------------------

import random
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]
random.shuffle(file_list)
#file_list = file_list[0:10]

print 'Creating test set...'
X_test, Y_test = load_patients(file_list[-TEST_SIZE:])
file_list = file_list[:-TEST_SIZE]

NUM_EPOCHS=30

print('Training...\n')
for i in range(NUM_EPOCHS):
    #random.shuffle(file_list)

    reader = IOReader(2, # number of items to be stored on the buffer
        data_loader,file_list=file_list) #function that reads the input and returns a generator
    reader.start() # start reader on background and fill the buffer

    for X_train, Y_train in reader.data_generator(): #(3): for testing purposes
            logging.info('Epoch: %d/%d' % (i, NUM_EPOCHS))
            logging.info('size: %s %s' % (str(X_train.shape), str(Y_train.shape)))
            model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE, 
                shuffle=True, callbacks=[tb])

    model.save(model_path + OUTPUT_MODEL)
    logging.info("Ending processing one epoch" )
    reader.join()


# print 'Training...'
# for i in range(NUM_EPOCHS):
#     random.shuffle(file_list)
#     X_train, Y_train = load_patients(file_list)
#     model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=BATCH_SIZE,
#                    shuffle=True, callbacks=[tb])  # validation_data=(X_test, Y_test),


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


# # TESTING NETWORK --------------------------------------------------
# from skimage import measure


# def get_regions(nodule_mask):
#     thr = np.where(nodule_mask < np.mean(nodule_mask), 0., 1.0)  # threshold detected regions
#     thr = np.where(nodule_mask < 0.8*np.max(nodule_mask), 0., 1.0)  # threshold detected regions
#     label_image = measure.label(thr)  # label them
#     labels = label_image.astype(int)
#     regions = measure.regionprops(labels, nodule_mask)
#     return regions

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


# from time import time
# mylist = os.listdir(input_path)
# file_list_dsb = [g for g in mylist if g.startswith('dsb_')]


# with open(wp + 'models/output_model_teixi.csv', 'a') as file:


#     for idx, filename in enumerate(file_list_dsb):
#         tstart = time()


#         b = np.load(os.path.join(input_path, filename))['arr_0']
#         X = []
#         print 'Patient: %s (%d/%d)' % (filename, idx, len(file_list_dsb))
#         for nslice in range(b.shape[1]):
#             if nslice%3 in [0,1]:
#                 continue
#             X = []
#             lung_image = b[0,nslice,:,:]
#             lung_mask = b[1,nslice,:,:]
#             lung_image[lung_mask==0]=-1000  # apply mask
#             X.append(normalize(lung_image))
#             X = np.expand_dims(np.asarray(X),axis=1)

#             pred = model.predict([X], verbose=0)

#             regions_pred = get_regions(pred[0,0])
#             for r in regions_pred:
#                 # print '%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,
#                 #                                            r.max_intensity, r.min_intensity, r.mean_intensity)

#                 file.write('%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,
#                                                            r.max_intensity, r.min_intensity, r.mean_intensity))
#         print time()-tstart




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
