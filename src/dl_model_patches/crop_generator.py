import os
from time import time
import logging
import multiprocessing
import numpy as np
from dl_model_patches import  common

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


wp = os.environ['LUNG_PATH']
INPUT_PATH = wp + 'data/preprocessed5_sample'
filenames = os.listdir(INPUT_PATH)
filenames = [os.path.join(INPUT_PATH, f) for f in filenames]


def load_and_store(filename):
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = common.load_patient(patient_data, output_rois=True, thickness=1)
    logging.info(stats)
    X = np.asarray(X)
    preds = model.predict(X, verbose=1)
    return rois, preds

pool = multiprocessing.Pool(4)
tstart = time()
rois, preds = zip(*pool.map(load_and_store, filenames[0:5]))
print "Total time:",time() - tstart

print len(rois)




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