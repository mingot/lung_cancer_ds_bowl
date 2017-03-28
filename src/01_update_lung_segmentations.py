# update lung segmentations from the specified
import os
import numpy as np
from time import time
from utils import lung_segmentation, plotting
import matplotlib.pyplot as plt


wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'  '/mnt/hd2/preprocessed5'
OUTPUT_PATH = '/mnt/hd2/preprocessed5_ls'
filenames = os.listdir(INPUT_PATH)

filenames = [
    'dsb_f1a64fda219db48bcfb8ad3823ef9fc1.npz',
    'dsb_402a18fa05523f80abb1e6d97793cc19.npz',
    'dsb_ca3cdb2e771df9989df5a9cb0296c44c.npz',
    'dsb_1e0f8048728717064645cb758eb89279.npz',
    'dsb_edad1a7e85b5443e0ae9e654d2adbcba.npz',
    'dsb_6e5f12931ef179cc21382a59f5acab86.npz',
    'dsb_11f10c2a0bfd231deeec98d69e4d0767.npz',
    'dsb_16377fe7caf072d882f234dbbff9ef6c.npz',
    'dsb_2969c7ad0e550fee1f4a68fcb3bbb9e5.npz',
    'dsb_cf0a772e90a14d77d664ce9baedf0e5c.npz',
    'dsb_91d0606b85ab7dbc7bab718c1312f2df.npz',
    'dsb_c87a713d17522698958de55c97654beb.npz'
]



## MULTIPROCESSING -------------------------------------------------------------------------------------------

import multiprocessing


def calc_segmentation(filename):
    print "Computing %s" % filename
    p = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
    p[1] = lung_segmentation.segment_lungs(p[0],fill_lung=True,method='thresholding2')
    np.savez_compressed(os.path.join(OUTPUT_PATH, filename), p)

pool = multiprocessing.Pool(4)
tstart = time()
pool.map(calc_segmentation, filenames)
print time() - tstart



## SINGLE THREAD -------------------------------------------------------------------------------------------


# for idx, filename in enumerate(filenames):
#     tstart00 = time()
#
#     print "Patient %s (%d/%d)" % (filename, idx, len(filenames))
#
#     tstart = time()
#     p = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
#     # print "Loading time:", time() - tstart
#
#     tstart = time()
#     new_lung = lung_segmentation.segment_lungs(p[0],fill_lung=True,method='thresholding2')
#     p[1] = new_lung
#     # print "Segmenting time:", time() - tstart
#
#     tstart = time()
#     np.savez_compressed(os.path.join(OUTPUT_PATH, filename), p)
#     # print "Storing time:", time() - tstart
#
#     print "Total time:", time() - tstart00

# p1 = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
# p2 = np.load(os.path.join(OUTPUT_PATH, filename))['arr_0']
#
#
# plt.imshow(p1[1,70])
# plt.show()
# plt.imshow(p2[1,70])
# plt.show()

