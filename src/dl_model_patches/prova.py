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
import multiprocessing



# PATHS
wp = os.environ['LUNG_PATH']
LUNA_ANNOTATIONS = wp + 'data/luna/annotations.csv'
OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
OUTPUT_MODEL = wp + 'personal/jm_patches_train_v06_local.hdf5'
INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = wp + 'data/preprocessed5_sample'



# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
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
#nodules_df = nodules_df[nodules_df['score'] > SCORE_TH]  # TODO: this filter should include the TN through the label
nodules_df['patientid'] = [f.split('/')[-1] for f in nodules_df['patientid']]  # TODO: remove when fixed the patient id without whole path
nodules_df['nslice'] = nodules_df['nslice'].astype(int)

# Construction of training and testsets
filenames_train = [os.path.join(INPUT_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(INPUT_PATH) and f in annotated]
filenames_test = [os.path.join(VALIDATION_PATH,f) for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(VALIDATION_PATH) and f in annotated]


def extract_regions_from_patient(patient, nodules_df):
    regions = []
    for idx, row in nodules_df.iterrows():
        x, y, d = int(row['x']), int(row['y']), int(row['diameter']+10)
        a = common.AuxRegion(bbox = [max(0,x-d/2), max(0,y-d/2), x+d/2, y+d/2])
        regions.append(a)
    return regions


def load_patient_with_candidates(patient_filename, thickness=0):
    """
    Provides the crops of the different images suggested at patient_nodules_df
    (ideally TP and FPs hard negatives).
    """
    patient_nodules_df = nodules_df[nodules_df['patientid']==patient_filename.split('/')[-1]]

    patient = np.load(patient_filename)['arr_0']
    nslices = list(set(patient_nodules_df['nslice']))


    logging.info("Loading patient: %s" % patient_filename)
    X, y = [], []
    for nslice in nslices:

        sel_patient_nodules_df = patient_nodules_df[patient_nodules_df['nslice']==nslice]
        regions_pred = extract_regions_from_patient(patient, sel_patient_nodules_df)
        regions_real = common.get_regions(patient[2,nslice], threshold=np.mean(patient[2,nslice]))
        labels, stats = common.get_labels_from_regions(regions_real, regions_pred)


        # TODO: remove when filtering good candidates is done in the begining
        idx_sel = [i for i in range(len(regions_pred)) if labels[i]==1 or sel_patient_nodules_df.iloc[i]['score']>SCORE_TH]
        regions_pred = [regions_pred[i] for i in idx_sel]
        labels = [labels[i] for i in idx_sel]

        lung_image = patient[0, nslice]
        if thickness>0:  # add extra images as channels for thick resnet
            lung_image = patient[0,(nslice - thickness):(nslice + thickness + 1),:,:]
            if lung_image.shape[0] != 2*thickness + 1:  # skip the extremes
                continue
        cropped_images = common.extract_crops_from_regions(img=lung_image, regions=regions_pred)


        X.extend(cropped_images)
        y.extend(labels)

    return X, y


print 'Loading files...'
tstart = time()
pool = multiprocessing.Pool(4)
result = zip(*pool.map(load_patient_with_candidates, filenames_train[0:10]))

# for filename in filenames_train[0:5]:
#     load_patient_with_candidates(filename)

print time() - tstart

#print result