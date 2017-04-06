# Read .csv containing detected nodules on LUNA, augment them with their
#   score (overlaping area with groundtruth) and image features

import numpy as np
from math import ceil
import pandas as pd
import scipy.misc as spm
from scipy import ndimage as ndi
import collections
import re #regex

# pca
import sklearn.decomposition as skd

import os
import pdb # debug
import multiprocessing as mp #parallel
from functools import partial

from dl_utils.heatmap import extract_regions_from_heatmap

# skimage
import skimage.segmentation as sks
import skimage.filters as skfi
# from skimage.draw import circle_perimeter
import skimage.morphology as skm
import skimage.measure as skme
from skimage.morphology import disk, square
import skimage.feature as skf
from skimage import measure
from skimage.feature import hog

# plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import plotting

## PATHS

# NODULES_FILE = wp + 'models/output_model_teixi_total_v2.csv'
# DATA_PATH = wp + 'data/preprocessed5_sample/'
# OUTPUT_FILE = wp + 'data/final_model/hog_v3_total.csv'
#NODULES_FILE = wp + 'output/noduls_unet_v02.csv'

# wp = os.environ['LUNG_PATH']
# DATA_PATH = '/mnt/hd2/preprocessed5/'
# NODULES_FILE = wp + 'output/noduls_thickunet_v01.csv'
# OUTPUT_FILE = wp + 'output/noduls_unet_v02_extended.csv'


# ## Load nodules from DL
# df_node = pd.read_csv(NODULES_FILE)
# df_node = df_node[(df_node['x']!=0) & (df_node['y']!=0) & (df_node['x']!=511) & (df_node['y']!=511)]  # filter

# # Load file list
# file_list = [g for g in os.listdir(DATA_PATH) if g.startswith('luna_')]

# BEGIN CSV AUGMENTATION -----------------------------------------------------------------

def extract_patch(
    df, 
    patient_id, 
    patient_path, 
    patient_colname='patientid', 
    out_size=(25, 25), 
    padding=1, # padding, deactivated by default
    swap_xy=False, 
    C=-500, 
    W=1500, 
    verbose=False):
    """
    This function extracts, from the deep learning data_frame and 
    a patient id, the following as an ordered dict: 
    (1) original patch, square 
    (2) original lung mask
    (3) resized patch
    (4) resized lung mask
    
    df: data_frame from deep learning
    patient_id: patient_code.npz to compute
    patient_path: path where the .npz files are stored
    out_size: size of the resized patches (None for original size)
    padding: function to add a padding to the diameter
    swap_xy: should x and y be swapped? Debug purposes
    C: center of filtering
    W: width of filtering
    verbose: show debug messages
    """
    # data frame for the patient and preprocessd file
    np_pat = np.load(patient_path + '/' + patient_id)['arr_0']
    df_pat = df[df[patient_colname] == patient_id]
    
    # compute volume of lung accumulated in the z axis
    sum_xy = np.sum(np_pat[1], axis=(1, 2))
    #perc_z = np.cumsum(sum_xy)/float(np.sum(sum_xy))
    
    patches = []
    for ind, row in df_pat.iterrows(): 
        z = int(row['nslice'])
        cx = int(row['x'])
        cy = int(row['y'])
        r = int(ceil(padding*(row['diameter'])/2.)) # now diameter is padded!
        
        if verbose: 
            print 'Slice: {} CX: {} CY: {}'.format(z, cx, cy)
        
        x = range(max(0, cx - r), min(511, cx + r + 1))
        y = range(max(0, cy - r), min(511, cy + r + 1))
    
        hu = np_pat[0, z]
        hu[hu < C - W/2] = C - W/2
        hu[hu > C + W/2] = C + W/2
        lung = np_pat[1, z]
        
        if swap_xy:
            img_hu = hu[np.ix_(y, x)]
            img_lung = lung[np.ix_(y, x)]
        else:
            img_hu = hu[np.ix_(x, y)]
            img_lung = lung[np.ix_(x, y)]
        
        # resizing is optional now
        if verbose:
            print 'Patch x: {} y: {} nslice: {} Original size: {}'.format(x, y, z, img_hu.shape)
        
        if out_size is None:
            if verbose:
                print 'Using original size...'
            if (img_hu.shape[0] < 15) | (img_hu.shape[1] < 15):
                out_size2 = (15, 15)
                if verbose:
                    print 'Patch is too small! Resized to (15, 15)'
            else:
                out_size2 = img_hu.shape
        else:
            out_size2 = out_size
        
        if verbose:
            print 'New size: {}'.format(out_size2)

        
        resc_hu = spm.imresize(img_hu, out_size2)
        resc_lung = spm.imresize(img_lung, out_size2, interp='lanczos')
        resc_lung = np.where(resc_lung > 255/2., 1.0, 0.0)
        
        # if all the pixels are lung, mask should be all ones
        # (this fails by default)
        if img_lung.astype(int).sum() == np.array(img_lung.shape).prod(): 
            resc_lung[:, :] = 1
        
        patches.append({'img_hu': img_hu, 
                        'img_lung': img_lung, 
                        'resc_hu': resc_hu, 
                        'resc_lung': resc_lung
        })
    
        if verbose:
            print "Patient patch appended. Next..."
    
    if verbose:
            print "Patient patches are processed! Next..."
    return patches, df_pat, sum_xy



def process_img(img, lung_mask, return_images=False): 
    """
    This function processes the given patch and returns the 
    skimage metrics properties object 
    (by now, largest segmented area)
    
    img: grayscale patch
    lung_mask: binary mask for lung pixels
    return_images: return images instead of regionprops (useful for plotting and debugging)
    """
    # 1 filtering
    #ex_1 = skfi.sobel(img)
    ex_1 = skfi.gaussian(img, sigma=1)
    # plt.imshow(ex_1)
    
    
    # 2 detect edges
    #ex_2 = sks.active_contour(ex_1, )
    #ex_2 = skf.canny(ex_1)
    
    # exception if image has only one colour
    try:
        thresh = skfi.threshold_otsu(ex_1)
    except TypeError: # blank image
        return None
        
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
    
    # apply lung mask
    ex_2_lung = ex_2*lung_mask
    
    # 3 fill regions
    ex_3 = ndi.binary_fill_holes(ex_2_lung)
    
    # plt.imshow(ex_3)
    
    # 4 opening
    ex_4 = skm.opening(ex_3, skm.square(3))
    # plt.imshow(ex_4)
    
    # 5 remove border
    #  this can clear everything if the node is ON the border
    # If we lose more than 40% of the area just undo it
    area_4 = ex_4.sum()
    if area_4 == 0:
        return None
    
    ex_5 = sks.clear_border(ex_4)
    area_5 = ex_5.sum()
    
    if float(area_4 - area_5)/area_4 > .4:
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
    ans['2b_lungmask'] = lung_mask
    ans['3_fillholes'] = ex_3
    ans['4_opening'] = ex_4
    ans['5_clearborder'] = ex_5
    ans['6_labelling'] = ex_6
    
    if return_images: 
        return ans
        
    # features
    ans_prop = skme.regionprops(ex_6, ex_1)
    
    # no region found?
    if len(ans_prop) == 0:
        return None
    
    # if labelling found several regions, we only take the largest for now
    ans_size = [p.area for p in ans_prop]
    ans_whichmax = ans_size.index(max(ans_size))
    ans_max = ans_prop[ans_whichmax]
    
    return ans_max
    # return ans, ans_prop



def process_prop(prop):
    """
    This function extracts some features from a scikit image
    properties object, as an ordered dict.
    
    prop: object from regionprops (skimage.measure._regionprops._RegionProperties)
    """
    if prop is None:
        return None
    
    hu_moments = prop.moments_hu
    hu_names = ['11_hu'+str(b) for b in np.arange(len(hu_moments))]
    
    # inertia
    eig0 = prop.inertia_tensor_eigvals[0]
    eig1 = prop.inertia_tensor_eigvals[1]
    
    # basic features
    # added 'esbeltez' and hu moments from Jacobo!
    # it was the samem as eccent.
    dict_ans = {
        '01_eccentricity': prop.eccentricity, 
        '02_extent': prop.extent, 
        '03_area': prop.area, 
        '04_perimeter': prop.perimeter, 
        '05_solidity': prop.solidity, 
        '06_mean_intensity': prop.mean_intensity, 
        '07_max_intensity': prop.max_intensity, 
        '08_inertia_tensor_eigvals0': eig0, 
        '09_inertia_tensor_eigvals1': eig1
    }
    dict_hu = dict(zip(hu_names, hu_moments))
    dict_ans.update(dict_hu)
    
    return pd.DataFrame.from_records([dict_ans])


def process_plot(list_dict):
    """
    Plot images from process_img in a grid. 
    Debug and inspection purposes.
    
    list_dict: patches from process_img. A list whose elements are the dictionaries
        generated with process_img
    """
    
    n_row = len(list_dict)
    n_col = len(list_dict[0])
    
    fig, axs = plt.subplots(
        nrows = n_row, ncols = n_col, figsize = (25, 25))
    plt.gray()
    # fig.subplots_adjust(hspace = .5, wspace=.001)
    #axs = axs.ravel()
    
    for ax_row, dict_plot in enumerate(list_dict): 
        for ax_col, (key, value) in enumerate(dict_plot.iteritems(), 0):
            axs[ax_row, ax_col].imshow(value)
            axs[ax_row, ax_col].set_title(key)
    
    plt.show()
    
    

def process_lbp(img_hu):
    """
    Compute LBPs and return the bins. 
    We have to check that all the bins are represented, because if the last ones
    are zeros and could be missing.
    
    img_hu: image (hu values) to extract LSBs from
    """
    ans = skf.local_binary_pattern(img_hu, P=8*3, R=3, method='uniform').astype(int)
    ans_bins = np.bincount(ans.ravel())
    ans_bins2 = ans_bins/float(sum(ans_bins))
    
    # complete zeros that could be missing
    ans_bins2 = np.r_[ans_bins2, np.zeros(26 - len(ans_bins))]
    
    return ans_bins2
    

def compress_feature(df, feature, n_components=3):
    REGEX = '[0-9]+_' + feature + '.+'
    
    # subset
    df_feat = df.filter(regex=REGEX)
    
    # PCA
    pca_feat = skd.PCA(n_components=n_components, copy=False)
    df_feat_pca = pca_feat.fit_transform(df_feat)
    
    # New colnames and new data frame
    names_feat = ['PC' + str(x) + '_' + feature for x in range(1, n_components+1)]
    df_feat_pca = pd.DataFrame(
        data = df_feat_pca, 
        index = df_feat.index, 
        columns = names_feat)
    
    # Drop old columns
    return pd.concat([df.select(lambda x: not re.search(REGEX, x), axis=1), df_feat_pca], axis=1)
  
    
# def process_pipeline_patient(**kwargs):
def process_pipeline_patient(
    patient_id, 
    df, 
    patient_path, 
    patient_colname='patientid',
    patient_inverted=[], 
    padding=1,
    out_size=(25, 25), 
    props_and_lbp='resc', 
    verbose=False):
    """
    This function processes a single patient from a data frame. 
    It is a wrapper to parallelise the code.
    
    patient_id: patient.npz to process
    df: whole data frame
    patient_path: path to find npy files
    patient_colname: name of the data frame column containing patients (should be patientid)
    patient_inverted: inverted patients as a list
    padding: padding factor for the diameter in patch extraction
    out_size: size of the resized patches (None for original size)
    props_and_lbp: use 'resc' rescaled image, or 'img' original image
    verbose: show debug messages
    """
    print 'Processing patient {} ...'.format(patient_id)
    # debug
    # pat = list_patient[0]
    
    # (1) Extract patchs from data frame and one patient
    if verbose:
        print "Extracting patches..."
    p_patch, p_df, sum_xy = extract_patch(
        df, 
        patient_id=patient_id, 
        patient_path=patient_path, 
        patient_colname=patient_colname,
        out_size=out_size, 
        padding=padding,
        swap_xy=False, 
        verbose=verbose)
    
    
    source_hu = props_and_lbp + '_hu'
    source_lung = props_and_lbp + '_lung'
    
    # (2) Extract properties (not features yet)
    if verbose:
        print "Extracting props..."
    p_prop = [process_img(img[source_hu], img[source_lung]) for img in p_patch]
    
    # (3.0) % of lung (differentiate walls from interior)
    if verbose:
        print "Extracting lungmask..."
    lungmask_feat = [float((img['img_lung']).sum())/(img['img_lung']).size for img in p_patch]
    lungmask_df = pd.DataFrame.from_records([{'10_lungmask':feat} for feat in lungmask_feat])
    
    # Extract meaningful features
    # TODO: also use (weighted?) hu moments, HOG, LBP, use lung mask in the process
    # this returns 1-row dfs for each patch, or None 
    
    # (3.1) HOG features
    if verbose:
        print "Extracting hogs..."
    hog_feat = [skf.hog(img['resc_hu'], pixels_per_cell=(10,10), cells_per_block=(2,2)) for img in p_patch]
    hog_names = ['20_hog'+str(b) for b in np.arange(len(hog_feat[0]))]
    hog_df = pd.DataFrame.from_records([dict(zip(hog_names, feat)) for feat in hog_feat])
    
    # (3.2) LBP for texture
    if verbose:
        print "Extracting lbps..."
    lbp_feat = [process_lbp(img[source_hu]) for img in p_patch]
    lbp_names = ['30_lbp'+str(b) for b in np.arange(len(lbp_feat[0]))]
    lbp_df = pd.DataFrame.from_records([dict(zip(lbp_names, lbp)) for lbp in lbp_feat])
    
    # (3.3) extract basic properties
    p_feat = [process_prop(p) for p in p_prop]
    
    # p_filtered = [x is None for x in p_feat]
    # p_all = zip(p_df, p_feat)[p_filtered]
    # df_all = pd.concat([p_df.iloc[[ind]].astype(dict).append(feat) for ind, feat in enumerate(p_feat) if feat is not None])
    
    # Combine all data frames
    # removed slices 
    
    # (4) indices of the non-null patches (some patches are null because 
    # segmentation in (2) did not find anything)
    patch_nonnull = [x is not None for x in p_feat]
    if np.array(patch_nonnull).sum() == 0:
        print 'None of the patches in patient {} were found any region'.format(patient_id)
        return None
    
    # data frame with features
    # (5) data_frame of features
    # pdb.set_trace()
    # data frame with features
    if verbose:
        print "Concatenating data frames..."
    df_feat = pd.concat(p_feat)
    df_feat.index = np.array(patch_nonnull).nonzero()[0]
    # concatenate all data frames (indices are a pain in the ass)
    df_augmented = pd.concat([
        df_feat, 
        hog_df.iloc[patch_nonnull], 
        lbp_df.iloc[patch_nonnull], 
        lungmask_df.iloc[patch_nonnull]], 
        axis=1)
    # keep track of original indices
    df_augmented.index = p_df.index[patch_nonnull]
    
    # recover indices
    # 
    
    df_original = p_df.iloc[patch_nonnull]
    
    # map position of the nodule (upper lobe, etc)
    # vertical positions of nodules
    z_nodule = df_original.nslice
    # first, compute accum volume in the z axis
    perc_z = np.cumsum(sum_xy)/float(np.sum(sum_xy))
    # reverse if patient is reversed
    if patient_id in patient_inverted:
        perc_z = 1 - perc_z
    # return percentage of lung volume before the nodule
    df_augmented['40_nodeverticalposition'] = [perc_z[int(x)] for x in z_nodule]
    
    # (6) concat data frames to obtain the final augmented data frame for this patient
    # df_all = pd.merge(p_df.iloc[patch_nonnull], df_feat, how='cross')
    df_all = pd.concat([df_original, df_augmented], axis=1)
    return df_all

def process_pipeline_csv(
    csv_in, 
    patient_path, 
    csv_out = None, 
    csv_as_files = True,
    dmin = 3, 
    dmax = 100, 
    compress={'hog':3, 'lbp':3, 'hu':2},
    nCores=1,
    patient_colname='patientid',
    patient_inverted=[], 
    padding=1,
    out_size=(25, 25), 
    props_and_lbp='resc',
    verbose=False):
    """
    This function creates an augmented features csv file
    
    csv_in: csv file from dl
    patient_path: path where the .npz files are stored
    csv_out: csv file to write, or None if result is a data frame
    csv_as_files: true if both input and output are csv filenames, false if they are data frames 
    dmin: minimum diameter (filtering)
    dmax: maximum diameter (filtering)
    compress: dictionary, features as keys and number of principal components as values
    nCores: number of cores to use 
    patient_colname: name of the patient column
    patient_inverted: inverted patients as a list (same format as their notation in the data frame)
    padding: padding factor for the diameter 
    out_size: size of the resized patches (None for original size)
    props_and_lbp: use 'resc' rescaled image, or 'img' original image
    verbose: show debug messages
    """
    # debug
    # csv_in='../data/tiny_dl_example.csv'
    # csv_out='dummy_out.csv'
    # patient_path="/home/sergi/all/devel/big/lung_cancer_ds_bowl/preprocessed5/"
    # verbose=False
    
    # Check format
    if csv_as_files:
        df_dl = pd.read_csv(csv_in)
        print 'Reading csv! Checking format is standard...'
    else:
        df_dl = csv_in
        print 'You have enterd data frame as input... function will return a data frame as well'
        
    df_dl_header = list(df_dl)
    for i in [patient_colname, 'x', 'y', 'diameter']:
        if not i in df_dl_header:
            print '{} is not in the header of the csv file. Aborting...'.format(i)
            raise ValueError('Header in csv file does not contain pipeline names.')
    print 'Success! Data frame shape: {}'.format(df_dl.shape)
    
    # filter bad patches
    df_dl_filter = df_dl[df_dl['x'].between(1, 510) & 
        df_dl['y'].between(1, 510) & 
        df_dl['diameter'].between(dmin, dmax)]
    print 'Filtering invalid patches. New shape: {}'.format(df_dl_filter.shape)
    print 'Filtered nodes with diameters outside ({} {})'.format(dmin, dmax)
    
    # different patients
    worker = mp.Pool(nCores)
    
    list_patient = df_dl_filter[patient_colname].unique()
    print 'Total of patients: {}'.format(len(list_patient))
    
    # list of data frames
    
    # this function has to be defined using partial
    # the only moving argument is patient_id
    # otherwise pool does not recognise it
    f_map = partial(
        process_pipeline_patient, 
        df = df_dl_filter, 
        patient_path = patient_path, 
        patient_colname=patient_colname,
        patient_inverted=patient_inverted,
        padding=padding,
        out_size=out_size,
        props_and_lbp=props_and_lbp,
        verbose=verbose)
            
        #     def f_map(pat):
        # process_pipeline_patient(
        #     df_dl_filter, 
        #     pat, 
        #     patient_path, 
        #     patient_colname='patientid',
        #     verbose=False)

    # dict_args = [{'df_dl': df_dl_filter, 
    #             'patient_id': pat, 
    #             'patient_path': patient_path,
    #             'patient_colname': 'patientid', 
    #             'verbose': verbose} for pat in list_patient]

    
    # parallel map function
    df_list = worker.map(f_map, list_patient)
    # df_out = []
    # for pat in list_patient:
    #     df_all = process_pipeline_patient(
    #         df_dl_filter, 
    #         pat, 
    #         patient_path, 
    #         patient_colname='patientid',
    #         verbose=False)
    #     df_out.append(df_all)
    
    df_list = pd.concat(df_list)
    if len(compress):
        for key, value in compress.iteritems():
            print 'Compressing feature: {} using {} PCs ...'.format(key, value)
            df_list = compress_feature(df_list, feature=key, n_components=value)
    
    print 'Filtered data frame shape: {}'.format(df_dl_filter.shape)
    print 'Final shape (rows whose nodes were not found were dropped): {}'.format(df_list.shape)
    
    if csv_as_files:
        print 'Done! Writing csv...'
        df_list.to_csv(csv_out, index=False)
    else:
        return df_list

# END CSV AUGMENTATION -----------------------------------------------------------------

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


# INDIVIDUAL CHECKS -----------------------------------------------------------------

# ## Checks for specific patient
# filename = "luna_126631670596873065041988320084.npz"
# patient = np.load(DATA_PATH + filename)['arr_0']
# for nslice in range(patient.shape[1]):
#     if patient[2,nslice].any()!=0:
#         print nslice
# nslice = 85
# visualize_csv(patient[0,nslice], df_node[(df_node['filename']==filename) & (df_node['nslice']==nslice)])
# plotting.plot_mask(patient[0,nslice], patient[2,nslice])
# plt.imshow(patient[1,nslice])
# plt.show()
#
# plotting.cube_show_slider(patient[0])
#
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




# FINAL CSV LOADING -----------------------------------------------------------------

## Generate features, score for each BB and store them
# Temporary commented this (Sergi)
# tp, fp, fn = 0, 0, 0
# with open(OUTPUT_FILE, 'w') as file:
#     for idx, filename in enumerate(file_list):  # to extract form .csv
#         #filename = "luna_126631670596873065041988320084.npz"

#         if filename[0:5]!="luna_":
#             continue

#         patient = np.load(DATA_PATH + filename)['arr_0']

#         print "Patient %s (%d/%d)" % (filename, idx, len(file_list))

#         if patient.shape[0]!=3:  # skip labels without ground truth
#             continue

#         slices = []
#         for nslice in range(patient.shape[1]):
#             if patient[2,nslice].any()!=0:
#                 slices.append(nslice)

#         for idx, row in df_node[df_node['filename']==filename].iterrows():
#             # row = df_node[(df_node['filename']==filename)].iloc[300]
#             cx = int(row['x'])  # row
#             cy = int(row['y'])  # column
#             z = int(row['nslice'])
#             r = int(ceil(row['diameter']/2.))

#             # # extract hog features
#             # img_hu = patient[0,z,(cx-r):(cx+r+1),(cy-r):(cy+r+1)]
#             # img_hu = 255.0*(img_hu - np.min(img_hu))/(np.max(img_hu) - np.min(img_hu))
#             # img_hu = img_hu.astype(np.int, copy=False)
#             # resc_hu = spm.imresize(img_hu, (20,20))
#             # features_hog, img  = hog(resc_hu, pixels_per_cell=(10,10),  cells_per_block=(2,2), visualise=True)
#             # ff = [str(f) for f in features_hog]
#             #
#             # plt.imshow(img_hu)
#             # plt.show()
#             #
#             # pp = process_img(img_hu)

#             # Get the score of the region (ground truth)
#             regions = extract_regions_from_heatmap(patient[2,z])
#             if len(regions)>1:
#                 print 'Patient: %s has more than 1 region at slice %d' % (filename, z)
#             a = AuxRegion([cx - r, cy - r, cx + r + 1, cy + r + 1])  # x1, y1, x2, y2
#             score = intersection_regions(a,regions[0])
#             if score>0.3:
#                 tp+=1
#                 if z in slices:
#                     slices.remove(z)
#             else:
#                 fp+=1

#             file.write("%.4f,%s,%d,%d,%d,%s\n" % (score, filename, z, cx, cy, ",".join(ff)))

#         fn += len(slices)
#         print "Results TP:%d FP:%d FN:%d of %d candidates" % (tp,fp,fn,len(df_node[df_node['filename']==filename].index))

