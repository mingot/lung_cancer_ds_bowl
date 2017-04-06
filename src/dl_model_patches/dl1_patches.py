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

OUTPUT_MODEL = wp + 'models/jm_patches_train_v19.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
LOGS_PATH = wp + 'logs/%s' % str('v19')
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')



### PATCHES GENERATION -----------------------------------------------------------------

# ## PATIENTS FILE LIST
# patients_with_annotations = pd.read_csv(NODULES_PATH)  # filter patients with no annotations to avoid having to read them
# patients_with_annotations = list(set(patients_with_annotations['seriesuid']))
# patients_with_annotations = ["luna_%s.npz" % p.split('.')[-1] for p in patients_with_annotations]
#
# filenames = os.listdir(INPUT_PATH)
# filenames = [g for g in filenames if g.startswith('luna_')]
# filenames_train = [os.path.join(INPUT_PATH, fp) for fp in filenames if fp in patients_with_annotations]
# filenames_test = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH) if fp in patients_with_annotations]
#
#
# def __load_and_store(filename):
#     patient_data = np.load(filename)['arr_0']
#     X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=True, output_rois=True, debug=True, include_ground_truth=True, thickness=1)
#     logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
#     return X, y, stats
#
#
#
# common.multiproc_crop_generator(filenames_train,
#                                 os.path.join(PATCHES_PATH,'dl1_v19_x_train.npz'),
#                                 os.path.join(PATCHES_PATH,'dl1_v19_y_train.npz'),
#                                 __load_and_store)
#
# common.multiproc_crop_generator(filenames_test,
#                                 os.path.join(PATCHES_PATH,'dl1_v19_x_test.npz'),
#                                 os.path.join(PATCHES_PATH,'dl1_v19_y_test.npz'),
#                                 __load_and_store)


### TRAINING -----------------------------------------------------------------

# Data augmentation generator
train_datagen = ImageDataGenerator(
    # rotation_range=30,  # .06,
    # width_shift_range=0.1, #0.02,
    # height_shift_range=0.1, #0.02,
    #shear_range=0.0002,
    #zoom_range=0.0002,
    dim_ordering="th",
    horizontal_flip=True,
    vertical_flip=True
    )

test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure


def chunks(X, y, batch_size=32, augmentation_times=4, thickness=0, is_training=True):
    """
    Batches generator for keras fit_generator. Returns batches of patches 40x40px
     - augmentation_times: number of time to return the data augmented
     - concurrent_patients: number of patients to load at the same time to add diversity
     - thickness: number of slices up and down to add as a channel to the patch
    """
    while 1:
        prct_pop, prct1 = 0.2, 0.2  # (1) of all the training set, how much we keep (2) % of 1's
        idx_1 = [i for i in range(len(y)) if y[i]==1]
        idx_1 = random.sample(idx_1, int(prct_pop*len(idx_1)))
        idx_0 = [i for i in range(len(y)) if y[i]==0]
        idx_0 = random.sample(idx_0, int(len(idx_1)/prct1))
        selected_samples = idx_0 + idx_1
        random.shuffle(selected_samples)
        logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y[selected_samples]), len(y[selected_samples])-sum(y[selected_samples])))

        # generator: if testing, do not augment data
        data_generator = train_datagen if is_training else test_datagen

        i, good = 0, 0
        lenX = len(selected_samples)
        for X_batch, y_batch in data_generator.flow(X[selected_samples], y[selected_samples], batch_size=batch_size, shuffle=is_training):
            i += 1
            if good*batch_size > lenX*augmentation_times or i>100:  # stop when we have augmented enough the batch
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                continue
            good += 1
            yield X_batch, y_batch

# LOADING PATCHES FROM DISK
logging.info("Loading training and test sets")
x_train = np.load(os.path.join(PATCHES_PATH, 'dl1_v19_x_train.npz'))['arr_0']
y_train = np.load(os.path.join(PATCHES_PATH, 'dl1_v19_y_train.npz'))['arr_0']
y_train = np.expand_dims(y_train, axis=1)
x_test = np.load(os.path.join(PATCHES_PATH, 'dl1_v19_x_test.npz'))['arr_0']
y_test = np.load(os.path.join(PATCHES_PATH, 'dl1_v19_y_test.npz'))['arr_0']
y_test = np.expand_dims(y_test, axis=1)
logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))


# Load model
model = ResnetBuilder().build_resnet_50((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# logging.info('Loading exiting model...')
# model.load_weights(OUTPUT_MODEL)


model.fit_generator(generator=chunks(x_train, y_train, batch_size=32, thickness=1),
                    samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=1600,
                    verbose=1,
                    #class_weight={0:1., 1:4.},
                    callbacks=[tb, model_checkpoint],
                    validation_data=chunks(x_test, y_test, batch_size=32, thickness=1, is_training=False,),
                    nb_val_samples=32*10,
                    max_q_size=10,
                    initial_epoch=1120,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers

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

#
# ## Tests data augmentation
# IF = wp + 'data/preprocessed5_sample/'
# filenames = [IF+f for f in os.listdir(IF)]
# filename = random.choice(filenames)
# patient_data = np.load(filename)['arr_0']
# X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, debug=True, thickness=1)
#
#
# train_datagen = ImageDataGenerator(
#     rotation_range=30,  # .06,
#     width_shift_range=0.1, #0.02,
#     height_shift_range=0.1, #0.02,
#     #shear_range=0.0002,
#     #zoom_range=0.0002,
#     dim_ordering="th",
#     horizontal_flip=True,
#     vertical_flip=True
#     )
#
# X = np.asarray(X)
# y = np.expand_dims(np.asarray(y), axis=1)
# train_datagen.fit(X[0:10])
#
# xres = []
# i = 0
# for Xp,yp in train_datagen.flow(X[0:1], y[0:1], batch_size=10, shuffle=True):
#     i+=1
#     if i>30:
#         break
#     xres.append(Xp)
#
# plotting.multiplot(X[0])
# plotting.multiplot([xres[i][0] for i in range(len(xres))])
#
#
#
# nslices = list(set([r[0] for r in rois]))
# nslice = random.choice(nslices)
# regions = [r[1] for r in rois if r[0]==nslice]
# plotting.plot_bb(patient_data[0,nslice], regions)
#
# # big ratio
# # 1.0/3<=ratio and ratio<=3
# idxs = [i for i in range(len(rois)) if common.calc_ratio(rois[i][1])>3 or common.calc_ratio(rois[i][1])<1.0/3]
# idxs = [i for i in range(len(rois)) if common.calc_area(rois[i][1])>70*70]
# idxs_2 = [i for i in range(len(y)) if y[i]==1]
#
# [i for i in idxs if i in idxs_2]
#
# idx = idxs[15]
# plotting.plot_bb(patient_data[0,rois[idx][0]], [rois[idx][1]])
# plotting.multiplot(X[idx])
