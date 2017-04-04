from __future__ import absolute_import, division
from keras.models import Model, load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Flatten, Dense, Activation, ZeroPadding2D, Dropout, Cropping2D
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import os
from time import time
from keras.callbacks import ModelCheckpoint
from dl_utils.tb_callback import TensorBoard
K.set_image_dim_ordering('th')
import random
import logging

from architecture import residual_unet_model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def weighted_loss(y_true, y_pred, pos_weight=100, epsilon=1e-5):
    #if this argument is greater than 1 we will penalize more the nodules not detected as nodules, we can set it up to 10 or 100?
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    y_pred_f = K.clip(y_pred_f, epsilon, 1-epsilon) #clipping away from 0 and 1 to avoid NAN in loss computation
    return K.mean(-(1-y_true_f)*K.log(1-y_pred_f)-y_true_f*K.log(y_pred_f)*pos_weight)

def zero_loss(y_true, y_pred, pos_weight=100, epsilon=1e-5):
    #if this argument is greater than 1 we will penalize more the nodules not detected as nodules, we can set it up to 10 or 100?
    y_true_f = K.flatten(y_true)  # y_true.flatten()
    y_pred_f = K.flatten(y_pred)  # y_pred.flatten()
    y_pred_f = K.clip(0*y_pred_f, epsilon, 1-epsilon) #clipping away from 0 and 1 to avoid NAN in loss computation
    return K.mean(-(1-y_true_f)*K.log(1-y_pred_f)-y_true_f*K.log(y_pred_f)*pos_weight)

class ThickRESUNET(object):
    def __init__(self, input_shape=(5,512,512), saved_file=None, pos_weight=100, lr=1e-5):

        self.thickness = input_shape[0]

        logging.info('Added thickness %d ' % self.thickness)

        self.model = residual_unet_model(input_shape)        
        self.model.compile(optimizer=Adam(lr=lr), loss=weighted_loss, metrics=[weighted_loss,zero_loss])

        if saved_file is not None:
            try:
                self.model.load_weights(saved_file)
            except:
                print 'EXPECTED MODEL'+self.model.get_config()
                print '-------------------'
                print 'SAVED MODEL'+load_model(saved_file, custom_objects={'weighted_loss': weighted_loss}).get_config()
                raise Exception("WARNING: the file does not contain a model matching this arquitecture!!")

    def train(self, 
              nb_epochs,  #number of epochs to train
              samples_per_epoch, #epoch is defined as a fixed number of samples
              batch_size, # batch size
              train_file_list, # list of training filenames
              validation_file_list=[], nb_val_samples=None, #validation data
              logs_path=None,
              modeloutput_path=None,
              max_q_size=10,  #length of queue
              nb_worker=1,    #number of parallel threads
              max_slices=0):   
        
        if logs_path is not None:
            # tensorboard logs
            logs_path = os.path.join(logs_path,'%s' % str(int(time())))
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
            modchk = ModelCheckpoint(modeloutput_path, save_best_only=True)
       
        if len(validation_file_list)>0 and nb_val_samples is None: # precomputation to provide keras with the number of samples in validation
            nb_val_samples = 0
            for x,y in chunks(validation_file_list,batch_size,infinite=False,thickness=self.thickness):
                nb_val_samples = nb_val_samples + x.shape[0]
            logging.info('Added %d validation samples' % nb_val_samples)
    
        
        if nb_val_samples is not None:
            self.model.fit_generator(generator=uniform_chunks(file_list=train_file_list,batch_size=batch_size,infinite=True,thickness=self.thickness,max_slices=max_slices),
                samples_per_epoch=samples_per_epoch,
                nb_epoch=nb_epochs,
                verbose=1,
                callbacks=[tb, modchk],
                validation_data=chunks(validation_file_list,batch_size,infinite=True,thickness=self.thickness),
                nb_val_samples=nb_val_samples,
                max_q_size=max_q_size,
                nb_worker=nb_worker)
        else:
            self.model.fit_generator(generator=chunks(train_file_list,batch_size,infinite=True,thickness=self.thickness,max_slices=max_slices),
                samples_per_epoch=samples_per_epoch,
                nb_epoch=nb_epochs,
                verbose=1,
                callbacks=[tb, modchk],
                max_q_size=max_q_size,
                nb_worker=nb_worker)
            

    def predict(self, in_file, batch_size):
        """ This function receives an input file path and outputs the matrix of expected probabilites of being a node"""
        Z = None
        for X,Y,slices in chunks([in_file], batch_size=batch_size, slices_data=True,infinite=False, full=True, thickness=self.thickness):
            out = self.model.predict_on_batch(X)
            len_slices = [len(x) for x in slices]
            if Z is None:
                Z = np.repeat(out,len_slices,axis=0)
            else:
                Z = np.concatenate((Z, np.repeat(out,len_slices,axis=0)),axis=0)
        
        return Z

    def _read_file(self,file_path, batch_len):
        pass

    def _read_files(self,file_list, infinite=False, len_batch=3):
        pass


# -----------------------------------------------------------------------------------------------    
    
def chunks(file_list=[], batch_size=5, infinite=True, thickness=5, slices_data=False,full=False,max_slices=0):
    while True:
        for filename in file_list:
            a, b, slices = load_patient(filename,thickness,full,max_slices=max_slices)
            if len(a.shape) != 4: # xapussa per evitar casos raros com luna_243094273518213382155770295147.npz que li passa a aquest???
                continue

            size = a.shape[0]
            nb_batch = int(np.ceil(size / float(batch_size)))
            for i in range(0,nb_batch):
                start = i * batch_size
                end = min(size, (i + 1) * batch_size)
                if slices_data is False:
                    yield a[start:end], b[start:end]
                else:
                    yield a[start:end], b[start:end], [slices[i] for i in range(start,end)]
        if infinite is False:
            break    

def uniform_chunks(batch_size=5, **kwargs):
    x = chunks(batch_size=batch_size, **kwargs)
    a, b = x.next()

    tmp_a = a[0:0]
    tmp_b = b[0:0]
    while True:
        a, b = x.next()

        if a.shape[0] == batch_size:
            yield a, b
        else:
            tmp_a = np.concatenate([tmp_a, a])
            tmp_b = np.concatenate([tmp_b, b])
            l_tmp = tmp_a.shape[0]

            if l_tmp >= batch_size:
                yield tmp_a[0:batch_size], tmp_b[0:batch_size]
                
                tmp_a = tmp_a[batch_size:l_tmp]
                tmp_b = tmp_b[batch_size:l_tmp]


#----------------------------------------------    
# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

# how a single file is loaded -------------------------------------------

def load_patient(filename,thickness=5, full=True,max_slices=0):

    logging.info('Added patient thickness %d ' % thickness)
    logging.info('Added patient max_slices %d ' % max_slices)


    X, Y = [], []
    b = np.load(filename)['arr_0']
    if b.shape[0]!=3: # si falta la nodule_mask asumim 0's
        x = b.shape[1:]
        b=np.concatenate(([b[0,:,:,:]],[b[1,:,:,:]],[np.zeros(x)]),axis=0)

    last_slice = -1e3  # big initialization
    slices = []
    max_slice=b.shape[1]
    d = (thickness-1)/2

    z_slices = range(0,max_slice,thickness)
    random.shuffle(z_slices)

    if max_slices>0:
        z_slices = z_slices[0:max_slices]

    for j in z_slices:
        zrange = range(j,min(j+thickness,max_slice))

        lung_image = b[0,zrange,:,:]
        lung_mask = b[1,zrange,:,:]
        nodules_mask = b[2,zrange,:,:]

        #lung_image[lung_mask==0]=-1000  # apply mask
        
        if len(zrange) != thickness: #we are in the last (partial layer) and we need to fill it
            last = lung_image[[-1]]
            lung_image = np.concatenate((lung_image, np.repeat(last,thickness-len(zrange),axis=0)), axis=0)
        
        # Discard if no nodules
        #if nodules_mask.sum() == 0:
        #    continue

        # Discard if bad segmentation
        voxel_volume_l = 2*0.7*0.7/(1000000.0)
        lung_volume_l = np.sum(lung_mask)/thickness*voxel_volume_l
        if (lung_volume_l < 0.02 or lung_volume_l > 0.1) and full is False:
            continue  # skip slices with bad lung segmentation

        # discard if consecutive slices
        #if j<last_slice + 5:
        #    continue

        # if ok append
        slices.append(zrange)

        X.append(normalize(lung_image))
        Y.append(np.max(nodules_mask,axis=0))  # nodules_mask
        #if len(slices)>5:  # at most 6 slices per patient
        #    break
    logging.info('patient %s added %d thickslices: %s' % (filename, len(slices), str(slices)))

    #shuffle the thickslices---------
    x = range(len(X))
    random.shuffle(x)

    X = [X[i] for i in x]
    Y = [Y[i] for i in x]
    # --------------------------------


    #X = np.expand_dims(np.asarray(X),axis=1)
    X = np.asarray(X)
    Y = np.expand_dims(np.asarray(Y),axis=1)
    return X, Y, slices
