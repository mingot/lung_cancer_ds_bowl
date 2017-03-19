import numpy as np
from math import ceil
import pandas as pd
import scipy.misc as spm
from scipy import ndimage as ndi
import collections
import matplotlib.pyplot as plt

import pdb # debug
import multiprocessing as mp #parallel
from functools import partial

# skimage
import skimage.segmentation as sks
import skimage.filters as skfi
# from skimage.draw import circle_perimeter
import skimage.morphology as skm
import skimage.measure as skme
from skimage.morphology import disk, square
import skimage.feature as skf

def extract_patch(
    df, 
    patient_id, 
    patient_path, 
    patient_colname='patientid', 
    out_size=(25, 25), 
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
    out_size: size of the resized patches
    swap_xy: should x and y be swapped? Debug purposes
    C: center of filtering
    W: width of filtering
    verbose: show debug messages
    """
    # data frame for the patient and preprocessd file
    np_pat = np.load(patient_path + '/' + patient_id)['arr_0']
    df_pat = df[df[patient_colname] == patient_id]
    
    patches = []
    for ind, row in df_pat.iterrows(): 
        z = int(row['nslice'])
        cx = int(row['x'])
        cy = int(row['y'])
        r = int(ceil(row['diameter']/2.))
        
        if verbose: 
            print 'Slice: {} CX: {} CY: {}'.format(z, cx, cy)
        
        x = range(max(0, cx - r), min(512, cx + r + 1))
        y = range(max(0, cy - r), min(512, cy + r + 1))
    
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
        
        
        resc_hu = spm.imresize(img_hu, out_size)
        resc_lung = spm.imresize(img_lung, out_size, interp='lanczos')
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
    return patches, df_pat



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
    dict_ans = {
        '01_eccentricity': prop.eccentricity, 
        '02_extent': prop.extent, 
        '03_area': prop.area, 
        '04_perimeter': prop.perimeter, 
        '05_solidity': prop.solidity, 
        '06_mean_intensity': prop.mean_intensity, 
        '07_max_intensity': prop.max_intensity, 
        '08_inertia_tensor_eigvals0': eig0, 
        '09_inertia_tensor_eigvals1': eig1, 
        '10_esb': np.sqrt(1 - eig1/eig0)
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
    
    # complete zeros that could be missing
    ans_bins = np.r_[ans_bins, np.zeros(26 - len(ans_bins))]
    
    return ans_bins
    
    
# def process_pipeline_patient(**kwargs):
def process_pipeline_patient(
    patient_id, 
    df, 
    patient_path, 
    patient_colname='patientid',
    verbose=False):
    """
    This function processes a single patient from a data frame. 
    It is a wrapper to parallelise the code.
    
    patient_id: patient.npz to process
    df: whole data frame
    patient_path: path to find npy files
    patient_colname: name of the data frame column containing patients (should be patientid)
    verbose: show debug messages
    """
    print 'Processing patient {} ...'.format(patient_id)
    # debug
    # pat = list_patient[0]
    
    # (1) Extract patchs from data frame and one patient
    p_patch, p_df = extract_patch(
        df, 
        patient_id=patient_id, 
        patient_path=patient_path, 
        patient_colname=patient_colname,
        swap_xy=False, 
        verbose=verbose)
    
    # (2) Extract properties (not features yet)
    p_prop = [process_img(img['resc_hu'], img['resc_lung']) for img in p_patch]
    # Extract meaningful features
    # TODO: also use (weighted?) hu moments, HOG, LBP, use lung mask in the process
    # this returns 1-row dfs for each patch, or None 
    
    # (3.1) HOG features
    hog_feat = [skf.hog(img['resc_hu'], pixels_per_cell=(10,10), cells_per_block=(2,2)) for img in p_patch]
    hog_names = ['20_hog'+str(b) for b in np.arange(len(hog_feat[0]))]
    hog_df = pd.DataFrame.from_records([dict(zip(hog_names, feat)) for feat in hog_feat])
    
    # (3.2) LBP for texture
    lbp_feat = [process_lbp(img['resc_hu']) for img in p_patch]
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
    df_feat = pd.concat(p_feat)
    df_feat.index = np.array(patch_nonnull).nonzero()[0]
    # concatenate all data frames (indices are a pain in the ass)
    df_augmented = pd.concat([
        df_feat, 
        hog_df.iloc[patch_nonnull], 
        lbp_df.iloc[patch_nonnull]], 
        axis=1)
    # keep track of original indices
    df_augmented.index = p_df.index[patch_nonnull]
    
    # recover indices
    # 
    
    
    # (6) concat data frames to obtain the final augmented data frame for this patient
    # df_all = pd.merge(p_df.iloc[patch_nonnull], df_feat, how='cross')
    df_all = pd.concat([p_df.iloc[patch_nonnull], df_augmented], axis=1)
    return df_all

def process_pipeline_csv(
    csv_in, 
    patient_path, 
    csv_out, 
    nCores=1,
    patient_colname='patientid',
    verbose=False):
    """
    This function creates an augmented features csv file
    
    csv_in: csv file from dl
    patient_path: path where the .npz files are stored
    csv_out: csv file to write
    nCores: number of cores to use 
    patient_colname: name of the patient column
    verbose: show debug messages
    """
    # debug
    # csv_in='../data/tiny_dl_example.csv'
    # csv_out='dummy_out.csv'
    # patient_path="/home/sergi/all/devel/big/lung_cancer_ds_bowl/preprocessed5/"
    # verbose=False
    
    # Check format
    df_dl = pd.read_csv(csv_in)
    print 'Reading csv! Checking format is standard...'
    df_dl_header = list(df_dl)
    for i in [patient_colname, 'x', 'y', 'diameter']:
        if not i in df_dl_header:
            print '{} is not in the header of the csv file. Aborting...'.format(i)
            raise ValueError('Header in csv file does not contain pipeline names.')
    print 'Success! Data frame shape: {}'.format(df_dl.shape)
    
    # filter bad patches
    df_dl_filter = df_dl[df_dl['x'].between(1, 510) & 
        df_dl['y'].between(1, 510) & 
        df_dl['diameter'].between(3.5, 28)]
    print 'Filtering invalid patches. New shape: {}'.format(df_dl_filter.shape)
    
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
        verbose=False)
            
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
    
    print 'Done! Writing csv...'
    pd.concat(df_list).to_csv(csv_out, index=False)
