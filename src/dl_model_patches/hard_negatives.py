import os
import random
import logging
import pandas as pd
import numpy as np
from time import time
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from dl_networks.sample_resnet import ResnetBuilder
from dl_model_patches import common
from dl_utils.tb_callback import TensorBoard
from utils import plotting
import matplotlib.pyplot as plt
from skimage import transform



# PATHS
wp = os.environ['LUNG_PATH']
LUNA_ANNOTATIONS = wp + 'data/luna/annotations.csv'
OUTPUT_DL1 =  wp + 'output/noduls_patches_v06.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
OUTPUT_MODEL =  wp + 'models/jm_patches_hardnegative_v01.hdf5'  # OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
INPUT_PATH = '/mnt/hd2/preprocessed5/' # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna/' # VALIDATION_PATH = wp + 'data/preprocessed5_sample'
LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

# luna annotated samples (do not train over the samples not annotated)
luna_df = pd.read_csv(LUNA_ANNOTATIONS)
annotated = list(set(['luna_%s.npz' % p.split('.')[-1] for p in luna_df['seriesuid']]))

# filter TP and FP of the suggested by DL1
SCORE_TH = 0.7
nodules_df = pd.read_csv(OUTPUT_DL1)
nodules_df = nodules_df[nodules_df['score'] > SCORE_TH]
nodules_df['patientid'] = [f.split('/')[-1] for f in nodules_df['patientid']]  # TODO: remove when fixed the patient id without whole path
nodules_df['nslice'] = nodules_df['nslice'].astype(int)

# Construction of training and testsets
filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(INPUT_PATH) and f in annotated]
filenames_test = [os.path.join(VALIDATION_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(VALIDATION_PATH) and f in annotated]




def extract_regions_from_patient(patient, nodules_df):
    regions = []
    for idx, row in nodules_df.iterrows():
        x, y, d = int(row['x']), int(row['y']), int(row['diameter']+10)
        a = common.AuxRegion(bbox = [x-d/2, y-d/2, x+d/2, y+d/2])
        regions.append(a)
    return regions


def load_patient_with_candidates(patient_filename, patient_nodules_df, thickness=0):
    """
    Provides the crops of the different images suggested at patient_nodules_df
    (ideally TP and FPs hard negatives).
    """
    patient = np.load(patient_filename)['arr_0']
    nslices = list(set(patient_nodules_df['nslice']))

    logging.info("Loading patient: %s" % patient_filename)

    X, y = [], []
    for nslice in nslices:
        regions_pred = extract_regions_from_patient(patient, patient_nodules_df[patient_nodules_df['nslice']==nslice])
        lung_image = patient[0, nslice]
        if thickness>0:  # add extra images as channels for thick resnet
            lung_image = patient[0,(nslice - thickness):(nslice + thickness + 1),:,:]
            if lung_image.shape[0] != 2*thickness + 1:  # skip the extremes
                continue
        cropped_images = common.extract_crops_from_regions(img=lung_image, regions=regions_pred)

        regions_real = common.get_regions(patient[2,nslice], threshold=np.mean(patient[2,nslice]))
        labels, stats = common.get_labels_from_regions(regions_real, regions_pred)

        X.extend(cropped_images)
        y.extend(labels)

    return X, y


# Data augmentation generator
train_datagen = ImageDataGenerator(dim_ordering="th", horizontal_flip=True, vertical_flip=True)
test_datagen = ImageDataGenerator(dim_ordering="th")  # dummy for testing to have the same structure


def chunk_generator(filenames, nodules_df, thickness=0, batch_size=32, is_training=True):
    while 1:

        X, y = [], []
        random.shuffle(filenames)
        for filename in filenames[0:10]:
            patientid = filename.split('/')[-1]
            X_single, y_single = load_patient_with_candidates(filename, nodules_df[nodules_df['patientid']==patientid], thickness=thickness)
            X.extend(X_single)
            y.extend(y_single)

        logging.info("Loaded batch of patients with %d/%d positives" % (np.sum(y), len(y)))
        idx_sel = [i for i in range(len(X)) if y[i]==1 or random.uniform(0,1) < 1.2*np.mean(y)]
        X = [X[i] for i in idx_sel]
        y = [y[i] for i in idx_sel]
        logging.info("Downsampled to %d/%d positives" % (np.sum(y), len(y)))

        # convert to np array and add extra axis (needed for keras)
        X, y = np.asarray(X), np.asarray(y)
        y = np.expand_dims(y, axis=1)
        if thickness==0:
            X = np.expand_dims(X, axis=1)

        # generator: if testing, do not augment data
        data_generator = train_datagen if is_training else test_datagen

        i = 0
        for X_batch, y_batch in data_generator.flow(X, y, batch_size=batch_size, shuffle=is_training):
            logging.info("Data augmentaton iteration %d" % i)
            if i*batch_size > len(X)*2:  # stop when we have augmented enough the batch
                #print 'leaving because augment'
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                #print 'continue because batch sixe'
                #print X_batch.shape, y_batch.shape
                continue
            i += 1
            yield X_batch, y_batch



### TRAINING -----------------------------------------------------------------



# Load model
model = ResnetBuilder().build_resnet_34((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
# if USE_EXISTING:
#     logging.info('Loading exiting model...')
#     model.load_weights(OUTPUT_MODEL)


model.fit_generator(generator=chunk_generator(filenames_train, nodules_df, batch_size=16, thickness=1),
                    samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=500,
                    verbose=1,
                    callbacks=[tb, model_checkpoint],
                    validation_data=chunk_generator(filenames_test, nodules_df, batch_size=16, thickness=1, is_training=False),
                    nb_val_samples=16*2,
                    max_q_size=64,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers


# # check generator
# for X,y in chunk_generator(filenames_train, nodules_df, batch_size=16):
#     print 'RESULT:', X.shape, y.shape