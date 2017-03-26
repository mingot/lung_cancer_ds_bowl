import os
import logging
import numpy as np
from time import time
from keras.optimizers import Adam
from dl_networks.sample_resnet import ResnetBuilder
from dl_model_patches import  common


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'
NODULES_PATH = wp + 'data/luna/annotations.csv'
OUTPUT_MODEL = wp + 'models/jm_patches_hardnegative_v01.hdf5'
OUTPUT_CSV = wp + 'output/noduls_patches_hardnegative_v01.csv'



## Params and filepaths
# NOTA: Cargando validation luna i dsb
THICKNESS = 1
write_method = 'w'
file_list = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH)]
file_list += [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH) if fp.startswith('dsb_')]


# ## if the OUTPUT_CSV file already exists, continue it
previous_filenames = set()
if os.path.exists(OUTPUT_CSV):
    write_method = 'a'
    with open(OUTPUT_CSV) as file:
        for l in file:
            previous_filenames.add(l.split(',')[0])


# Load model
model = ResnetBuilder().build_resnet_34((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
logging.info('Loading existing model...')
model.load_weights(OUTPUT_MODEL)

with open(OUTPUT_CSV, write_method) as file:

    # write the header if the file is new
    if write_method=='w':
        file.write('patientid,nslice,x,y,diameter,score,label\n')

    for idx, filename in enumerate(file_list):
        if filename in previous_filenames:
            continue

        tstart = time()
        logging.info("Patient %s (%d/%d)" % (filename, idx, len(file_list)))
        try:
            patient_data = np.load(filename)['arr_0']
            X, y, rois, stats = common.load_patient(patient_data, output_rois=True, thickness=THICKNESS)

            if len(X)==0:
                continue

            X = np.asarray(X)
            if THICKNESS==0:
                X = np.expand_dims(X, axis=1)
            preds = model.predict(X, verbose=1)
        except:
            logging.info("Error in patient %s, skipping" % filename)
            continue

        for i in range(len(preds)):
            nslice, r = rois[i]
            file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (filename.split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],y[i]))

            # if preds[i]>0.8:
            #     logging.info("++ Good candidate found with (nslice,x,y,diam,score): %d,%d,%d,%.2f,%.2f" % (nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i]))

        logging.info("Total ROIS:%d, Good Candidates:%d, Time processnig:%.2f" % (len(preds), len([p for p in preds if p>0.8]), time()-tstart))



# # file_list = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH)]
# file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)][0:5]
# # file_list = [g for g in file_list if g.startswith('dsb_')]
#
# # Load model
# model = ResnetBuilder().build_resnet_34((3,40,40),1)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# logging.info('Loading exiting model...')
# model.load_weights(OUTPUT_MODEL)
#
# def load_and_store(filename):
#     patient_data = np.load(filename)['arr_0']
#     X, y, rois, stats = common.load_patient(patient_data, output_rois=True, thickness=1)
#     logging.info(stats)
#     X = np.asarray(X)
#     preds = model.predict(X, verbose=1)
#     return rois, preds
#
# pool = multiprocessing.Pool(4)
# tstart = time()
# rois, preds = zip(*pool.map(load_and_store, file_list[0:5]))
# print "Total time:",time() - tstart
#
#
# with open(wp + 'output/parallel_test.csv', 'w') as file:
#     for i in range(len(file_list)):
#         nslice, r = rois[i]
#         file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (file_list[i].split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i], preds[i]))