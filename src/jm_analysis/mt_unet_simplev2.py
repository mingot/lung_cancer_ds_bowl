import os
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from networks.unet_simplified import UNETArchitecture
from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler
from skimage import measure

K.set_image_dim_ordering('th')

# PARAMETERS
NUM_EPOCHS = 15
BATCH_SIZE = 40
USE_EXISTING = False  # load previous model to continue training
OUTPUT_MODEL = 'teixi_unet_simplified_v5_p1_detect_image.hdf5'


## paths
#wp = os.environ['LUNG_PATH']
model_path  = '/home/mteixido/models/'
input_path = '/mnt/hd2/preprocessed4'
#input_path = wp + '/preprocessed3' #/mnt/hd2/preprocessed2'
# input_path = '/mnt/hd2/preprocessed4'
logs_path = '/home/mteixido/logs/%s' % str(int(time()))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


# tensorboard logs
tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard


# DATA LOADING AND NORMALIZATION -----------------------------------------------------------------

# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


# we load one patient at a time but we shuffle the slices
import random
mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('luna_')]
random.shuffle(file_list)


def load_patients(filelist=file_list[0:3]):
    for filename in filelist:
        X, Y = [], []
        b = np.load(os.path.join(input_path, filename))['arr_0']
        for j in range(b.shape[1]):

            lung_image = b[0,j,:,:]
            lung_mask = b[1,j,:,:]
            
            lung_image = normalize(lung_image)

            X.append(lung_image)
            Y.append(lung_mask) #(nodules_mask)
        
    
        p=np.random.permutation(len(X)) # generate permutation of slices
        X = (np.asarray(X)[p,:,:])[0:2,:,:] #apply permutation
        Y = (np.asarray(Y)[p,:,:])[0:2,:,:] #apply permutation
        X = np.expand_dims(X,axis=1)
        Y = np.expand_dims(Y,axis=1)
        yield X, Y



# create model ----------------------------------------

## definition of dice_coef_loss 
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
#model.compile(optimizer=Adam(lr=0.1), loss=dice_coef_loss, metrics=[dice_coef_loss])
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['binary_crossentropy'])



# Async data loader (this should be improved and go to a separate file) --------------------------------------------------
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
        logging.info('IOReader started')
        for x in self.f():  # TODO modify the constructor to allow pass more arguments to f
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



# DATA LOADING AND TRAINING -------------------------------------------

def data_loader():
    for x in load_patients([file_list[0]]):
        yield x


# TRAINING NETWORK --------------------------------------------------

print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    #random.shuffle(file_list)

    reader = IOReader(2, # number of items to be stored on the buffer
        data_loader) #function that reads the input and returns a generator
    reader.start() # start reader on background and fill the buffer

    
    while True:
        try:
            print 'Epoch: %d/%d, new patient' % (i, NUM_EPOCHS)
            X_train, Y_train = reader.get()
            logging.info('trainer: Got data from IOReader')
            model.fit(X_train, Y_train, verbose=1, nb_epoch=500, batch_size=BATCH_SIZE, shuffle=True, callbacks=[tb])
        except StopIteration:
            break

    model.save(model_path + OUTPUT_MODEL)
    logging.info("Ending processing one epoch" )
    reader.join()
    


