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



### LOADING DATA -----------------------------------------------------------------

# Data augmentation generator
train_datagen = ImageDataGenerator(
    #rotation_range=.5,  # .06,
    #width_shift_range=0.05, #0.02,
    #height_shift_range=0.05, #0.02,
    #shear_range=0.0002,
    #zoom_range=0.0002,
    dim_ordering="th",
    horizontal_flip=True,
    vertical_flip=True
    )

test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure



def chunks(X_orig, y_orig, file_list=[], batch_size=32, augmentation_times=4, concurrent_patients=10, thickness=0, is_training=True):
    """
    Batches generator for keras fit_generator. Returns batches of patches 40x40px
     - augmentation_times: number of time to return the data augmented
     - concurrent_patients: number of patients to load at the same time to add diversity
     - thickness: number of slices up and down to add as a channel to the patch
    """
    while 1:
        # for j in range(0,len(file_list),concurrent_patients):
        #     filenames = file_list[j:(j+concurrent_patients)]
        #     X, y = [], []
        #     for filename in filenames:
        #         X_single, y_single = load_patient(filename, thickness=thickness)
        #         if len(X_single)==0:
        #             continue
        #         X.extend(X_single)
        #         y.extend(y_single)

        # downsample negatives (reduce 90%)
        selected_samples  = [i for i in range(len(y_orig)) if y_orig[i]==1 or random.randint(0,9)==0]
        X = [X_orig[i] for i in selected_samples]
        y = [y_orig[i] for i in selected_samples]
        logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y), len(y)-sum(y)))

        # convert to np array and add extra axis (needed for keras)
        X, y = np.asarray(X), np.asarray(y)
        y = np.expand_dims(y, axis=1)
        if thickness==0:
            X = np.expand_dims(X, axis=1)

        # generator: if testing, do not augment data
        data_generator = train_datagen if is_training else test_datagen

        i = 0
        for X_batch, y_batch in data_generator.flow(X, y, batch_size=batch_size, shuffle=True):
            i += 1
            if i*len(X_batch) > len(X)*augmentation_times:  # stop when we have augmented enough the batch
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                continue
            yield X_batch, y_batch



### MODEL LOADING -----------------------------------------------------------------


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'
NODULES_PATH = wp + 'data/luna/annotations.csv'
OUTPUT_MODEL = wp + 'models/jm_patches_train_v07.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
OUTPUT_CSV = wp + 'output/noduls_patches_v07.csv'
PATCHES_PATH = '/mnt/hd2/patches'  # PATCHES_PATH = wp + 'data/preprocessed5_patches'
LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
# tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
# model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
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
# file_list = os.listdir(INPUT_PATH)
# file_list = [g for g in file_list if g.startswith('luna_')]
# file_list_train = [os.path.join(INPUT_PATH, fp) for fp in file_list if fp in patients_with_annotations]
# file_list_test = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH) if fp in patients_with_annotations]
#
#
# ## STORING PATCHES IN DISK
# print "Generating and saving training set..."
# tstart, total_stats = time(), {}
# X_train, y_train = [], []
# for idx,filename in enumerate(file_list_train):
#     patientid = filename.split('/')[-1]
#     logging.info("Loading patient %s %d/%d" % (patientid, idx,len(file_list_train)))
#     patient_data = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
#     X_single, y_single, rois, stats = common.load_patient(patient_data, discard_empty_nodules=True, output_rois=True, debug=True, thickness=1)
#     total_stats = common.add_stats(stats, total_stats)
#     X_train.extend(X_single)
#     y_train.extend(y_single)
# print "Time generating: %.2f, Total stats: %s" % (time() - tstart, str(total_stats))
# print "Saving file..."
# np.savez_compressed(os.path.join(PATCHES_PATH,'x_train.npz'), np.asarray(X_train))
# np.savez_compressed(os.path.join(PATCHES_PATH,'y_train.npz'), np.asarray(y_train))
#
#
# print "Generating and saving test set..."
# tstart, total_stats = time(), {}
# X_test, y_test = [], []
# for idx,filename in enumerate(file_list_test):
#     patientid = filename.split('/')[-1]
#     logging.info("Loading patient %s %d/%d" % (patientid, idx,len(file_list_train)))
#     patient_data = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
#     X_single, y_single, rois, stats = common.load_patient(patient_data, discard_empty_nodules=True, output_rois=True, debug=True, thickness=1)
#     total_stats = common.add_stats(stats, total_stats)
#     X_test.extend(X_single)
#     y_test.extend(y_single)
# print "Time generating: %.2f, Total stats: %s" % (time() - tstart, str(total_stats))
# print "Saving file..."
# np.savez_compressed(os.path.join(PATCHES_PATH,'x_test.npz'), np.asarray(X_test))
# np.savez_compressed(os.path.join(PATCHES_PATH,'y_test.npz'), np.asarray(y_test))

### TRAINING -----------------------------------------------------------------

# # LOADING PATCHES FROM DISK
# logging.info("Loading training and test sets")
# x_train = np.load(os.path.join(PATCHES_PATH, 'x_train.npz'))['arr_0']
# y_train = np.load(os.path.join(PATCHES_PATH, 'y_train.npz'))['arr_0']
# x_test = np.load(os.path.join(PATCHES_PATH, 'x_test.npz'))['arr_0']
# y_test = np.load(os.path.join(PATCHES_PATH, 'y_test.npz'))['arr_0']
# logging.info("Training set (1s/total): %d/%d" % (sum(y_train),len(y_train)))
# logging.info("Test set (1s/total): %d/%d" % (sum(y_test), len(y_test)))
#
#
# # Load model
# model = ResnetBuilder().build_resnet_34((3,40,40),1)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# # logging.info('Loading exiting model...')
# # model.load_weights(OUTPUT_MODEL)
#
# model.fit_generator(generator=chunks(x_train, y_train, batch_size=32, thickness=1),
#                     samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
#                     nb_epoch=500,
#                     verbose=1,
#                     class_weight={0:1., 1:4.},
#                     callbacks=[tb, model_checkpoint],
#                     validation_data=chunks(x_test, y_test, batch_size=32, thickness=1, is_training=False),
#                     nb_val_samples=len(y_test),
#                     max_q_size=64,
#                     nb_worker=1)  # a locker is needed if increased the number of parallel workers

# ## CHECKS GENERATOR
# for i in range(10):
#     X, y = next(chunks(file_list_train[1:3], batch_size=4, thickness=1))
#     print X.shape, y.shape


### EVALUATE -----------------------------------------------------------------


# ## Params and filepaths
# THICKNESS = 1
# write_method = 'w'
# file_list = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH)]
# file_list += [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)]
#
# file_list = [g for g in file_list if g.startswith('dsb_')]
#
#
# ## if the OUTPUT_CSV file already exists, continue it
# previous_filenames = set()
# if os.path.exists(OUTPUT_CSV):
#     write_method = 'a'
#     with open(OUTPUT_CSV) as file:
#         for l in file:
#             previous_filenames.add(l.split(',')[0])


# with open(OUTPUT_CSV, write_method) as file:
#
#     # write the header if the file is new
#     if write_method=='w':
#         file.write('patientid,nslice,x,y,diameter,score,label\n')
#
#     for idx, filename in enumerate(file_list):
#         if filename in previous_filenames:
#             continue
#
#         logging.info("Patient %s (%d/%d)" % (filename, idx, len(file_list)))
#         try:
#             X, y, rois = common.load_patient(filename, output_rois=True, thickness=THICKNESS)
#
#             if len(X)==0:
#                 continue
#
#             X = np.asarray(X)
#             if THICKNESS==0:
#                 X = np.expand_dims(X, axis=1)
#             preds = model.predict(X, verbose=1)
#         except:
#             logging.info("Error in patient %s, skipping" % filename)
#             continue
#
#         for i in range(len(preds)):
#             nslice, r = rois[i]
#             file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (filename.split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],y[i]))
#
#             if preds[i]>0.8:
#                 logging.info("++ Good candidate found with (nslice,x,y,diam,score): %d,%d,%d,%.2f,%.2f" % (nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i]))


# file_list = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH)]
file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)][0:5]
# file_list = [g for g in file_list if g.startswith('dsb_')]

# Load model
model = ResnetBuilder().build_resnet_34((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
logging.info('Loading exiting model...')
model.load_weights(OUTPUT_MODEL)

def load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = common.load_patient(patient_data, output_rois=True, thickness=1)
    logging.info(stats)
    X = np.asarray(X)
    preds = model.predict(X, verbose=1)
    return rois, preds

pool = multiprocessing.Pool(4)
tstart = time()
rois, preds = zip(*pool.map(load_and_store, file_list[0:5]))
print "Total time:",time() - tstart


with open(wp + 'output/parallel_test.csv', 'w') as file:
    for i in range(len(file_list)):
        nslice, r = rois[i]
        file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (file_list[i].split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i], preds[i]))

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



# ### Quality checks for ROIs detection
# INPUT_PATH = wp + 'data/preprocessed5_sample'
# INPUT_PATH = wp + 'data/preprocessed5_sample_watershed'
# INPUT_PATH = wp + 'data/preprocessed5_sample_th2'
# file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)]
#
# total_stats = {}
# for filename in file_list:
#     X, y, rois, stats = load_patient(filename, discard_empty_nodules=True, output_rois=True, thickness=0)
#     print stats
#     total_stats = add_stats(stats, total_stats)
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
