import os
import random
import logging
import multiprocessing
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from utils import plotting
from dl_model_patches import  common
from sklearn import metrics
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'
NODULES_PATH = wp + 'data/luna/annotations.csv'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'

OUTPUT_MODEL = wp + 'models/jm_patches_train_v17.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % str('v17')

#LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)



# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
#tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
#model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


### PATCHES GENERATION -----------------------------------------------------------------

## PATIENTS FILE LIST
patients_with_annotations = pd.read_csv(NODULES_PATH)  # filter patients with no annotations to avoid having to read them
patients_with_annotations = list(set(patients_with_annotations['seriesuid']))
patients_with_annotations = ["luna_%s.npz" % p.split('.')[-1] for p in patients_with_annotations]

filenames = os.listdir(INPUT_PATH)
filenames = [g for g in filenames if g.startswith('luna_')]
filenames_train = [os.path.join(INPUT_PATH, fp) for fp in filenames if fp in patients_with_annotations]
filenames_test = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH) if fp in patients_with_annotations]


def __load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=True, output_rois=True, debug=True, thickness=1)
    logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
    return X, y, stats


# TODO: INCLOURE TOTES LES REGIONS POSITIVES TAMBE!!

common.multiproc_crop_generator(filenames_train[0:100],
                                os.path.join(PATCHES_PATH,'x_train_dl1_5.npz'),
                                os.path.join(PATCHES_PATH,'y_train_dl1_5.npz'),
                                __load_and_store, store=False)

# common.multiproc_crop_generator(filenames_test,
#                                 os.path.join(PATCHES_PATH,'x_test_dl1_5.npz'),
#                                 os.path.join(PATCHES_PATH,'y_test_dl1_5.npz'),
#                                 __load_and_store)


### TRAINING -----------------------------------------------------------------

# # Data augmentation generator
# train_datagen = ImageDataGenerator(
#     #rotation_range=10,  # .06,
#     #width_shift_range=0.05, #0.02,
#     #height_shift_range=0.05, #0.02,
#     #shear_range=0.0002,
#     #zoom_range=0.0002,
#     dim_ordering="th",
#     horizontal_flip=True,
#     vertical_flip=True
#     )
#
# test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure
#
#
# def chunks(X_orig, y_orig, batch_size=32, augmentation_times=4, thickness=0, is_training=True):
#     """
#     Batches generator for keras fit_generator. Returns batches of patches 40x40px
#      - augmentation_times: number of time to return the data augmented
#      - concurrent_patients: number of patients to load at the same time to add diversity
#      - thickness: number of slices up and down to add as a channel to the patch
#     """
#     while 1:
#         # downsample negatives (reduce 90%)
#         # if is_training:
#         #     idx_1 = [i for i in range(len(y_orig)) if y_orig[i]==1]
#         #     idx_0 = [i for i in range(len(y_orig)) if y_orig[i]==0]
#         #     idx_0 = random.sample(idx_0, len(idx_1))
#         #     selected_samples = idx_0 + idx_1
#         # else:
#
#         selected_samples  = [i for i in range(len(y_orig)) if y_orig[i]==1 or random.randint(0,9)==0]
#         X = [X_orig[i] for i in selected_samples]
#         y = [y_orig[i] for i in selected_samples]
#         logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y), len(y)-sum(y)))
#
#         # convert to np array and add extra axis (needed for keras)
#         X, y = np.asarray(X), np.asarray(y)
#         y = np.expand_dims(y, axis=1)
#         if thickness==0:
#             X = np.expand_dims(X, axis=1)
#
#         # generator: if testing, do not augment data
#         data_generator = train_datagen if is_training else test_datagen
#
#         i, good = 0, 0
#         for X_batch, y_batch in data_generator.flow(X, y, batch_size=batch_size, shuffle=is_training):
#             i += 1
#             if good*batch_size > len(X)*augmentation_times or i>100:  # stop when we have augmented enough the batch
#                 break
#             if X_batch.shape[0] != batch_size:  # ensure correct batch size
#                 continue
#             good += 1
#             yield X_batch, y_batch
#
#
# def chunks_multichannel(X_orig, y_orig, batch_size=32, augmentation_times=4, thickness=0, is_training=True):
#     """
#     Batches generator for keras fit_generator. Returns batches of patches 40x40px
#      - augmentation_times: number of time to return the data augmented
#      - concurrent_patients: number of patients to load at the same time to add diversity
#      - thickness: number of slices up and down to add as a channel to the patch
#     """
#     while 1:
#         # downsample negatives (reduce 90%)
#         if is_training:
#             len1 = int(0.25*batch_size)
#             idx_1 = [i for i in range(len(y_orig)) if y_orig[i]==1]
#             idx_1 = random.sample(idx_1, len1)
#             idx_0 = [i for i in range(len(y_orig)) if y_orig[i]==0]
#             idx_0 = random.sample(idx_0, batch_size - len1)
#             selected_samples = idx_0 + idx_1
#             random.shuffle(selected_samples)
#         else:
#             selected_samples = range(len(y_orig))
#
#         #selected_samples  = [i for i in range(len(y_orig)) if y_orig[i]==1 or random.randint(0,9)==0]
#         X = [X_orig[i] for i in selected_samples]
#         y = [y_orig[i] for i in selected_samples]
#         logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y), len(y)-sum(y)))
#
#         # convert to np array and add extra axis (needed for keras)
#         X, y = np.asarray(X), np.asarray(y)
#         y = np.expand_dims(y, axis=1)
#         if thickness==0:
#             X = np.expand_dims(X, axis=1)
#
#         yield X, y
#
# # LOADING PATCHES FROM DISK
# logging.info("Loading training and test sets")
# x_train = np.load(os.path.join(PATCHES_PATH, 'x_train_dl1_5.npz'))['arr_0']
# y_train = np.load(os.path.join(PATCHES_PATH, 'y_train_dl1_5.npz'))['arr_0']
# x_test = np.load(os.path.join(PATCHES_PATH, 'x_test_dl1_5.npz'))['arr_0']
# y_test = np.load(os.path.join(PATCHES_PATH, 'y_test_dl1_5.npz'))['arr_0']
# logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
# logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))
#
#
#
# # Load model
# model = ResnetBuilder().build_resnet_34((5,40,40),1)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# # logging.info('Loading exiting model...')
# # model.load_weights(OUTPUT_MODEL)
#
#
# model.fit_generator(generator=chunks_multichannel(x_train, y_train, batch_size=32, thickness=1),
#                     samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
#                     nb_epoch=500,
#                     verbose=1,
#                     #class_weight={0:1., 1:4.},
#                     callbacks=[tb, model_checkpoint],
#                     validation_data=chunks_multichannel(x_test, y_test, batch_size=32, thickness=1),  # TODO: is_training=False
#                     nb_val_samples=32*10,
#                     max_q_size=64,
#                     nb_worker=1)  # a locker is needed if increased the number of parallel workers

# ## CHECKS GENERATOR
# for i in range(10):
#     #X, y = next(chunks(x_test,y_test, batch_size=32, thickness=2))
#     X, y = next(chunks_multichannel(x_test,y_test, batch_size=32, thickness=2))
#     print X.shape, y.shape, np.mean(y)





### CHECKS -----------------------------------------------------------------


# ## Checking predictions
# filename = file_list[2]
# b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
# X, y, rois = load_patient(filename, discard_empty_nodules=False, output_rois=True, thickness=THICKNESS)
#
# # select slice to make prediction
# for j in range(b.shape[1]):
#     if np.sum(b[2,j])!=0:
#         print j
# nslice = 128
# plotting.plot_mask(b[0,nslice], b[2,nslice])
#
# sel_rois, sel_idx = [], []
# for idx,roi in enumerate(rois):
#     nslice_sel, r = roi
#     if nslice_sel == nslice:
#         sel_rois.append(r)
#         sel_idx.append(idx)
#
# sel_X = [X[i] for i in range(len(X)) if i in sel_idx]
#
# # make predictions
# preds = model.predict(np.asarray(sel_X), verbose=1)
# preds
#
# plotting.plot_bb(b[0,nslice], sel_rois)
# scored_rois = [sel_rois[i] for i in range(len(sel_rois)) if preds[i]>0.9]
# plotting.plot_bb(b[0,nslice], )



# # ### Quality checks for ROIs detection
# from skimage import morphology, measure
# def get_new_mask(pat_data):
#     # based on kernel: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
#     # remove the biggest princpial components to remove FPs
#     mask = pat_data[0].copy()
#     mask[pat_data[1]!=1] = -2000  # fuera de los pulmones a 0
#     mask[mask<-500] = -2000  # regiones de tejido no nodular/vaso, fuera
#
#     binary = morphology.closing(mask, morphology.ball(2))
#     binary[binary!=-2000] = 1
#     binary[binary==-2000] = 0
#
#     # nslice = 96
#     # plotting.plot_mask(mask[nslice], pat_data[2,nslice])
#     # plotting.plot_mask(binary[nslice], pat_data[2,nslice])
#
#     label_scan = measure.label(binary)
#     areas = [r.area for r in measure.regionprops(label_scan)]
#     areas.sort()
#
#     for r in measure.regionprops(label_scan):
#         max_x, max_y, max_z = 0, 0, 0
#         min_x, min_y, min_z = 1000, 1000, 1000
#
#         for c in r.coords:
#             max_z = max(c[0], max_z)
#             max_y = max(c[1], max_y)
#             max_x = max(c[2], max_x)
#
#             min_z = min(c[0], min_z)
#             min_y = min(c[1], min_y)
#             min_x = min(c[2], min_x)
#         if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
#             for c in r.coords:
#                 binary[c[0], c[1], c[2]] = 0
#         else:
#             index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
#     binary  = morphology.dilation(morphology.dilation(binary))
#     return binary
#
#
#
# INPUT_PATH = wp + 'data/preprocessed5_sample'
# #INPUT_PATH = wp + 'data/preprocessed5_sample_watershed'
# #INPUT_PATH = wp + 'data/preprocessed5_sample_th2'
# file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)]
#
# total_stats = {}
# for filename in file_list:
#     pat_data = np.load(filename)['arr_0']
#     #new_mask = get_new_mask(pat_data)
#     #pat_data[1] = new_mask
#     X, y, rois, stats = common.load_patient(pat_data, discard_empty_nodules=True, output_rois=True, thickness=0)
#     print stats
#     total_stats = common.add_stats(stats, total_stats)
#     print "TOTAL STATS:", filename.split('/')[-1], total_stats
#
# filename = 'luna_127965161564033605177803085629.npz'
# p = np.load(os.path.join(INPUT_PATH,filename))['arr_0']
# X, y, rois, stats = load_patient(os.path.join(INPUT_PATH,filename), discard_empty_nodules=True, output_rois=True, thickness=0)
# print stats
#
# nslice = 80
# regions = [r[1] for r in rois if r[0]==nslice]
# lung_image, lung_mask = p[0,nslice], p[1,nslice]
# plotting.plot_mask(p[0,nslice], p[2,nslice])
# plotting.plot_bb(p[0,nslice], regions)
#
# plotting.plot_mask(p[0,nslice], p[1,nslice])
# plt.imshow(p[1,nslice])
# plt.show()



# ## Calculate area regions of luna
# for idx, filename in enumerate(file_list):
#     b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
#     if b.shape[0]!=3:
#         continue
#
#     print 'Loading %s (%d/%d)' % (filename, idx, len(file_list))
#     for j in range(b.shape[1]):
#         if np.sum(b[2,j])!=0:
#             regions = get_regions(b[2,j])
#             for region in regions:
#                 print "Filename %s, slice %d, area %s" % (filename, j, str(calc_area(region)))




# ### Individual prediction checks
# plotting.multiplot(X[2300])
# b = np.load(INPUT_PATH+'/'+filename)['arr_0']
# for j in range(b.shape[1]):
#     if np.sum(b[2,j])!=0:
#         print j
#
# sel_nslice = 96
# sel_regions = []
# sel_ids = []
# for idx,r in enumerate(rois):
#     nslice, region = r
#     if nslice==sel_nslice:
#         sel_regions.append(region)
#         sel_ids.append(idx)
#
#
# plotting.plot_mask(b[0,sel_nslice], b[2,sel_nslice])
# plotting.plot_bb(b[0,sel_nslice], sel_regions[2])
#
# sel_ids[2]
# plotting.multiplot(X[4145])
# preds[4145]
#
# new_X = X[4145]
# new_X = np.expand_dims(new_X, axis=0)
# model.predict(new_X, verbose=1)
#
# #select biggest
# max_area = [0,0]
# for idx, region in enumerate(sel_regions):
#     if calc_area(region)>1500:
#         print idx, calc_area(region)


# ### Performance test
# import random
# file_list = os.listdir(INPUT_PATH)
# random.shuffle(file_list)
# NUM = 5
#
# tstart = time()
# for i in range(NUM):
#     X, y, rois = load_patient(file_list[i], discard_empty_nodules=False, output_rois=True, thickness=THICKNESS)
# print (time() - tstart)/NUM
