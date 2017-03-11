import numpy as np
from math import ceil
import pandas as pd
import scipy.misc as spm
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
        
        x = range(cx - r, cx + r + 1)
        y = range(cy - r, cy + r + 1)
    
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


import collections
import skimage.segmentation as sks
import skimage.filters as skfi
# from skimage.draw import circle_perimeter
from scipy import ndimage as ndi
import skimage.morphology as skm
import skimage.measure as skme
from skimage.morphology import disk, square
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




import collections
import skimage.segmentation as sks
import skimage.filters as skfi
# from skimage.draw import circle_perimeter
from scipy import ndimage as ndi
import skimage.morphology as skm
import skimage.measure as skme
def process_prop(prop):
    """
    This function extracts some features from a scikit image
    properties object, as an ordered dict.
    
    prop: object from regionprops (skimage.measure._regionprops._RegionProperties)
    """
    if prop is None:
        return None
        
    return pd.DataFrame.from_records([{
        '01_eccentricity': prop.eccentricity, 
        '02_extent': prop.extent, 
        '03_area': prop.area, 
        '04_perimeter': prop.perimeter, 
        '05_solidity': prop.solidity, 
        '06_mean_intensity': prop.mean_intensity, 
        '07_max_intensity': prop.max_intensity, 
        '08_inertia_tensor_eigvals0': prop.inertia_tensor_eigvals[0], 
        '09_inertia_tensor_eigvals1': prop.inertia_tensor_eigvals[1]
    }])

import matplotlib.pyplot as plt
def process_plot(list_dict):
    """
    Plot images from process_img in a grid
    
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