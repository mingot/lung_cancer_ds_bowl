# Read .csv containing detected nodules on LUNA, augment them with their
#   score (overlaping area with groundtruth) and image features

import pandas as pd
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.segmentation as sks
import skimage.filters as skfi
import skimage.morphology as skm
import skimage.measure as skme
import scipy.misc as spm
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import disk, square
from skimage.feature import hog
from math import ceil
# from utils import plotting


## PATHS
wp = os.environ['LUNG_PATH']
# NODULES_FILE = wp + 'models/output_model_teixi_total_v2.csv'
# DATA_PATH = wp + 'data/preprocessed5_sample/'
# OUTPUT_FILE = wp + 'data/final_model/hog_v3_total.csv'
DATA_PATH = '/mnt/hd2/preprocessed5/'
NODULES_FILE = wp + 'output/noduls_unet_v02.csv'
OUTPUT_FILE = wp + 'output/noduls_unet_v02_extended.csv'


## Load nodules from DL
df_node = pd.read_csv(NODULES_FILE)
df_node = df_node[(df_node['x']!=0) & (df_node['y']!=0) & (df_node['x']!=511) & (df_node['y']!=511)]  # filter

# Load file list
file_list = [g for g in os.listdir(DATA_PATH) if g.startswith('luna_')]



def get_regions(mask):
    # thr = np.where(nodule_mask < np.mean(nodule_mask), 0., 1.0)  # threshold detected regions
    thr = np.where(mask < np.mean(mask), 0., 1.0)  # threshold detected regions
    labels = measure.label(thr)  # label them
    regions = measure.regionprops(labels)
    return regions

def process_img(img):
    """
    This function processes the given patch and returns the
    skimage metrics properties object
    (by now, largest segmented area)

    img: grayscale patch
    """
    # 1 filtering
    #ex_1 = skfi.sobel(img)
    ex_1 = skfi.gaussian(img, sigma=1)
    # plt.imshow(ex_1)


    # 2 detect edges
    #ex_2 = sks.active_contour(ex_1, )
    #ex_2 = skf.canny(ex_1)
    thresh = skfi.threshold_otsu(ex_1)
    ex_2 = skm.closing(ex_1 > thresh, square(3))

    # plt.imshow(ex_2)

    # s = np.linspace(0, 2*np.pi,25)
    # init = 10*np.array([np.cos(s), np.sin(s)]).T+12.5
    # snake = sks.active_contour(ex, init, alpha=1e-3, beta=0.4)

    # plot the snake
    # axes = plt.gca()
    # axes.set_xlim([0, 24])
    # axes.set_ylim([0, 24])
    # plt.plot(snake[:, 0], snake[:, 1], linewidth = 3.0, color = "r")
    # plt.imshow(ex)

    # 3 fill regions
    ex_3 = ndi.binary_fill_holes(ex_2)

    # plt.imshow(ex_3)

    # 4 opening
    ex_4 = skm.opening(ex_3, skm.square(3))
    # plt.imshow(ex_4)

    # 5 remove border
    #  this can clear everything if the node is ON the border
    ex_5 = sks.clear_border(ex_4)
    if ex_5.sum() == 0:
        ex_5 = ex_4
    # plt.imshow(ex_5)

    # 6 labelling
    ex_6 = skme.label(ex_5)

    # 7 properties
    # props = skme.regionprops(ex_6, ex_1)
    # len(props)
    # props = props[0]
    # for prop in props:
    #     print(prop, props[prop])
    # props.area
    # props.inertia_tensor_eigvals

    # ex_6.sum()

    # props[0].area

    # 8 raw features
    # feat_lbp = skf.local_binary_pattern(ex_1, P=8, R=3, method='uniform')
    # feat_lbp.shape
    # plt.imshow(feat_lbp)
    # plt.hist(feat_lbp.ravel())
    ans = collections.OrderedDict()
    ans['0_raw'] = img
    ans['1_filter'] = ex_1
    ans['2_otsu'] = ex_2
    ans['3_fillholes'] = ex_3
    ans['4_opening'] = ex_4
    ans['5_clearborder'] = ex_5
    ans['6_labelling'] = ex_6

    # features
    ans_prop = skme.regionprops(ex_6, ex_1)

    # if labelling found several regions, we only take the largest for now
    ans_size = [p.area for p in ans_prop]
    ans_whichmax = ans_size.index(max(ans_size))
    ans_max = ans_prop[ans_whichmax]

    return ans_max
    # return ans, ans_prop

class AuxRegion():
    def __init__(self, dim):
        self.bbox = dim

def intersection_regions(r1, r2):
    h = min(r1.bbox[2], r2.bbox[2]) - max(r1.bbox[0], r2.bbox[0])
    w = min(r1.bbox[3], r2.bbox[3]) - max(r1.bbox[1], r2.bbox[1])
    intersectionArea = w*h
    if h<0 or w<0:
        return 0.0

    area1 = (r1.bbox[2] - r1.bbox[0])*(r1.bbox[3] - r1.bbox[1])
    area2 = (r2.bbox[2] - r2.bbox[0])*(r2.bbox[3] - r2.bbox[1])
    unionArea = area1 + area2 - intersectionArea
    overlapArea = intersectionArea*1.0/unionArea # This should be greater than 0.5 to consider it as a valid detection.
    return overlapArea

def visualize_csv(img, node_df):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for idx,row in node_df.iterrows():
        cx = row['x']  # row
        cy = row['y']  # column
        z = row['nslice']
        r = int(ceil(row['diameter']/2.))
        minr, minc, maxr, maxc = cx-r, cy-r, cx+r+1, cy+r+1
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()


# ## Checks for specific patient
# filename = "luna_126631670596873065041988320084.npz"
# patient = np.load(DATA_PATH + filename)['arr_0']
# for nslice in range(patient.shape[1]):
#     if patient[2,nslice].any()!=0:
#         print nslice
# nslice = 47
# visualize_csv(patient[0,nslice], df_node[(df_node['filename']==filename) & (df_node['nslice']==nslice)])
# plt.imshow(patient[1,nslice])
# plt.show()
# plotting.plot_mask(patient[0,nslice], patient[2,nslice])
#
# df_node[(df_node['filename']==filename)]
#
# plt.imshow(patient[2,54])
# plt.show()
#
# plotting.cube_show_slider(patient[0])

# # check afected slices per patient
# for filename in file_list:
#         patient = np.load(DATA_PATH + filename)['arr_0']
#         patient.shape
#
#         if patient.shape[0]!=3:  # skip labels without groundtruth
#             print 'patient %s no nodules' % filename
#             continue
#
#         slices = []
#         for nslice in range(patient.shape[1]):
#             if patient[2,nslice].any()!=0:
#                 slices.append(nslice)
#         print "patient %s slices: %s" % (filename, str(slices))

tp, fp, fn = 0, 0, 0
with open(OUTPUT_FILE, 'w') as file:
    for idx, filename in enumerate(file_list):  # to extract form .csv
        #filename = "luna_126631670596873065041988320084.npz"

        if filename[0:5]!="luna_":
            continue

        patient = np.load(DATA_PATH + filename)['arr_0']

        print "Patient %s (%d/%d)" % (filename, idx, len(file_list))

        if patient.shape[0]!=3:  # skip labels without ground truth
            continue

        slices = []
        for nslice in range(patient.shape[1]):
            if patient[2,nslice].any()!=0:
                slices.append(nslice)

        for idx, row in df_node[df_node['filename']==filename].iterrows():
            #row = df_node[df_node['filename']==filename].iloc[0]
            cx = row['x']  # row
            cy = row['y']  # column
            z = row['nslice']
            r = int(ceil(row['diameter']/2.))

            # extract hog features
            img_hu = patient[0,z,(cx-r):(cx+r+1),(cy-r):(cy+r+1)]
            img_hu = 255.0*(img_hu - np.min(img_hu))/(np.max(img_hu) - np.min(img_hu))
            img_hu = img_hu.astype(np.int, copy=False)
            resc_hu = spm.imresize(img_hu, (20,20))
            features_hog = hog(resc_hu, pixels_per_cell=(10,10),  cells_per_block=(2,2))
            ff = [str(f) for f in features_hog]

            # Get the score of the region (ground truth)
            regions = get_regions(patient[2,row['nslice']])
            if len(regions)>1:
                print 'Patient: %s has more than 1 region at slice %d' % (filename, row['nslice'])
            a = AuxRegion([cx - r, cy - r, cx + r + 1, cy + r + 1])  # x1, y1, x2, y2
            score = intersection_regions(a,regions[0])
            if score>0.3:
                tp+=1
                if z in slices:
                    slices.remove(z)
            else:
                fp+=1

            file.write("%.4f,%s,%d,%d,%d,%s\n" % (score, filename, z, cx, cy, ",".join(ff)))

        fn += len(slices)
        print "Results TP:%d FP:%d FN:%d of %d candidates" % (tp,fp,fn,len(df_node[df_node['filename']==filename].index))

