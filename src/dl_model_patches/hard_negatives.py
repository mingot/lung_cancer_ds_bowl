import os
import random
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from dl_networks.sample_resnet import ResnetBuilder
from dl_model_patches import common
from utils import plotting

K.set_image_dim_ordering('th')

wp = os.environ['LUNG_PATH']
OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv' # wp + 'output/noduls_patches_v06.csv'
INPUT_PATH = wp + 'data/preprocessed5_sample'

luna_df = pd.read_csv(wp + 'data/luna/annotations.csv')
annotated = ['luna_%s.npz' % p.split('.')[-1] for p in  luna_df['seriesuid']]

SCORE_TH = 0.7
nodules_df = pd.read_csv(OUTPUT_DL1)
nodules_df = nodules_df[nodules_df['score'] > SCORE_TH]
nodules_df['patientid'] = [f.split('/')[-1] for f in nodules_df['patientid']]  # TODO: remove when fixed
filenames  = [f for f in set(nodules_df['patientid']) if f[0:4]=='luna' and f in os.listdir(INPUT_PATH) and f in annotated]
nodules_df['nslice'] = nodules_df['nslice'].astype(int)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

def extract_regions_from_patient(patient, nodules_df):
    regions = []
    for idx, row in nodules_df.iterrows():
        x, y, d = int(row['x']), int(row['y']), int(row['diameter']+10)
        a = common.AuxRegion(bbox = [x-d/2, y-d/2, x+d/2, y+d/2])
        regions.append(a)
    return regions


plotting.plot_bb(patient[0,nslice], regions_pred)
plotting.plot_bb(patient[0,nslice], regions_real)
for j in range(patient.shape[1]):
    if np.sum(patient[2,j])!=0:
        print j


def load_patient(patient_filename, patient_nodules_df, thickness=0):
    patient = np.load(os.path.join(INPUT_PATH,patient_filename))['arr_0']
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
        for filename in filenames[0:4]:
            X_single, y_single = load_patient(filename, nodules_df[nodules_df['patientid']==filename], thickness=thickness)
            X.extend(X_single)
            y.extend(y_single)

        idx_sel = [i for i in range(len(X)) if y[i]==1 or random.uniform(0,1) < np.mean(y)]
        X = [X[i] for i in idx_sel]
        y = [y[i] for i in idx_sel]

        # convert to np array and add extra axis (needed for keras)
        X, y = np.asarray(X), np.asarray(y)
        y = np.expand_dims(y, axis=1)
        if thickness==0:
            X = np.expand_dims(X, axis=1)

        # generator: if testing, do not augment data
        data_generator = train_datagen if is_training else test_datagen

        i = 0
        for X_batch, y_batch in data_generator.flow(X, y, batch_size=batch_size, shuffle=is_training):
            if i*len(X_batch) > len(X)*4:  # stop when we have augmented enough the batch
                print 'leaving because augment'
                break
            if X_batch.shape[0] != batch_size:  # ensure correct batch size
                print 'continue because batch sixe'
                print X_batch.shape, y_batch.shape
                continue
            i += 1
            yield X_batch, y_batch



for X,y in chunk_generator(filenames, nodules_df, batch_size=8):
    print 'RESULT:', X.shape, y.shape


plt.imshow(X[13])
plt.show()

res = transform.rotate(X[13], 45, mode='reflect')

plt.imshow(res)
plt.show()


#### TRINAING ---------------------------

# Load model
model = ResnetBuilder().build_resnet_34((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# if USE_EXISTING:
#     logging.info('Loading exiting model...')
#     model.load_weights(OUTPUT_MODEL)


model.fit_generator(generator=chunk_generator(filenames, nodules_df, batch_size=16, thickness=1),
                    samples_per_epoch=160, #1280,  # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=500,
                    verbose=1,
                    #callbacks=[tb, model_checkpoint],
                    validation_data=chunk_generator(filenames, nodules_df, batch_size=32, thickness=1, is_training=False),
                    nb_val_samples=16*2,
                    max_q_size=64,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers