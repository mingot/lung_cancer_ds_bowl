import copy
import os
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from numpy.random import binomial

K.set_image_dim_ordering('th')

## paths
wp = os.environ['LUNG_PATH']
model_path  = wp + 'models/'
custom_dataset_path = wp + 'src/jc_dl/experiments/slice_classification/new_dataset/custom_dataset.npz'




#   JM_SLOWUNET
import os
import numpy as np
import math
from time import time
from keras.optimizers import Adam
from keras import backend as K
from networks.sample_cnn import Sample2DCNNNetworkArchitecture
#from utils.tb_callback import TensorBoard
from keras.callbacks import LearningRateScheduler

K.set_image_dim_ordering('th')

# PARAMETERS
NUM_EPOCHS = 20
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
#tb = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

print 'creating model...'
arch = Sample2DCNNNetworkArchitecture((1,512,512),False)
lrate = LearningRateScheduler(step_decay)
model = arch.get_model()
model.compile(optimizer=Adam(lr=1.0e-3), loss='binary_crossentropy', metrics=[])

if USE_EXISTING:
    print 'loading model...'
    model.load_weights(model_path + 'jc_samplecnn_v0.hdf5')


## Load LUNA dataset
# Load dataset
data = np.load(custom_dataset_path)
X_train, Y_train, X_test, Y_test = data['X_train'], data['Y_train'], data['X_test'], data['Y_test']
normalize = lambda x, mean_, std_: (x - mean_) / std_
X_test = normalize(X_test, X_train.mean(), X_train.std())
X_train = normalize(X_train, X_train.mean(), X_train.std())

X_train = np.expand_dims(np.asarray(X_train),axis=1)
X_test  = np.expand_dims(np.asarray(X_test ),axis=1)

print('Training...\n')
# model_checkpoint = keras.callbacks.ModelCheckpoint(model_path + 'jm_slowunet_v3.hdf5', monitor='loss', save_best_only=True)
for i in range(NUM_EPOCHS):
    #random.shuffle(file_list)
    #for j in range(43):
    #    print 'Epoch: %d/%d, batch:%d' % (i, NUM_EPOCHS, j*20)
    #    #X_train, Y_train = load_patients(file_list[j*20:(j+1)*20])
    model.fit(X_train, Y_train, verbose=1, nb_epoch=1, batch_size=20, validation_data=(X_test, Y_test), shuffle=True)#, callbacks=[tb])
    model.save(model_path + 'jc_samplecnn_v0.hdf5')


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
from pylab import *
def plot_matrix(i, case):
    subplot(3,3,1+i)
    imshow(X_train[i,0], cmap = cm.Greys)
figure()
[plot_matrix(i,x) for i,x in enumerate( [i for i in range(len(Y_train)) if Y_train[i] == 1][:9])]
figure()
[plot_matrix(i,x) for i,x in enumerate( [i for i in range(len(Y_train)) if Y_train[i] == 0][:9])]
show()