# update lung segmentations from the specified
import os
import numpy as np
from time import time
from utils import lung_segmentation, plotting
import matplotlib.pyplot as plt


wp = os.environ['LUNG_PATH']
# INPUT_PATH = wp + 'data/preprocessed5_sample'  # INPUT_PATH = wp + 'data/preprocessed5_sample'  '/mnt/hd2/preprocessed5'
# OUTPUT_PATH =  wp + 'data/preprocessed5_sample_th2'
INPUT_PATH = '/mnt/hd2/preprocessed5_wrong_lung_segmentation'
OUTPUT_PATH = '/mnt/hd2/preprocessed6_wrong_lung_segmentation'
filenames = os.listdir(INPUT_PATH)

#
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


# p1 = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
# p2 = np.load(os.path.join(OUTPUT_PATH, filename))['arr_0']
#
#
# plt.imshow(p1[1,70])
# plt.show()
# plt.imshow(p2[1,70])
# plt.show()

