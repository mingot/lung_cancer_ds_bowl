  import os, csv
import numpy as np
import SimpleITK as sitk
from glob import glob
from utils.reading import *
from utils.plotting import *
from utils.preprocessing import *
from scipy.ndimage.morphology import *
from skimage import filters
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from scipy.signal import periodogram, find_peaks_cwt
from scipy.stats import describe
from multiprocessing import Pool

import matplotlib.pyplot as plt

DEBUG = True
SERVER = os.uname()[1] == 'ip-172-31-7-211'

if SERVER:
    path = '/home/shared/data/stage1'
    preprocessed = '/mnt/hd2/preprocessed5'
    output_file = '/home/shared/data/stage1_extra_features_sex_predictors.csv'
else:
    path = '/home/carlos/DSB2017/dsb_sample'
    preprocessed = '/home/carlos/DSB2017/dsb_preprocessed'
    output_file = '/home/carlos/lung_cancer_ds_bowl/data/stage1_extra_features_sex_predictors_simple.csv'

patient_files = os.listdir(path)
patient_files = sorted(patient_files)

patient_files = ['4d86e1657d46b9ee44c2c434fad231ce', '579dfed23d49414a2c080bbf9d7f958f', '5fd33ea74e1ad740a201ae9b3c383fc5', '8269e2eb69b3ba38c48dd5ae6039e250']

common_spacing = [2., 0.7, 0.7]

def __3d_sobel__(pix):
    edges = np.array([ scipy.ndimage.filters.sobel(z_pix) for z_pix in pix ])
    print edges.shape
    scipy.ndimage.morphology.binary_closing(edges, iterations=1)
    cube_show_slider(edges)
    return edges

    
def segment_bones(image):
#    image = __remove_tube_mask__(image)
    
    lower_half = [y > image.shape[1]/2 for y in range(image.shape[1])]
    middle = [ 2*image.shape[2]/5 < x < 3*image.shape[2]/5 for x in range(image.shape[2]) ]
    mask = np.einsum('j,k->jk', lower_half, middle)
    bone = np.logical_and(image > 350, image < 2500)
    bone = np.einsum('ijk,jk->ijk', bone, mask)
    bone = scipy.ndimage.morphology.binary_dilation(bone, iterations=3)
    '''
    plt.imshow(scipy.ndimage.filters.sobel(bone[50,:,:]))
    plt.show()
    '''
    #__3d_sobel__(bone)
    return bone 

def get_mass_center(pix):
    '''returns mass center in voxel units'''
    tot = np.sum(pix)
    sz = np.einsum('ijk,i->', pix, range(pix.shape[0]))
    sy = np.einsum('ijk,j->', pix, range(pix.shape[1]))
    sx = np.einsum('ijk,k->', pix, range(pix.shape[2]))
    return np.array([sz,sy,sx]) / tot

def __get_weighted_var__(indices, y, f, avg):
    n = len(np.where(y[indices] != 0)[0])
    if n > 1:
        return n * 1. / (n-1) * np.sum(y[indices]*(1./f[indices] - avg)**2) / np.sum(y[indices])
    else:
        return 0.

def __get_lung_height__(lung_mask):
    exists_mask = np.where(np.sum(lung_mask, axis=(1,2)) > 0)
    if exists_mask[0].size == 0:
        return 0
    else: 
        return np.max(exists_mask) - np.min(exists_mask)


def get_intercostal_dist(pix, bone_mask, new_spacing, lung_height):
    '''
    bone = np.logical_and(pix > 350, pix < 2500)
    lower_half = [y > pix.shape[1]/2 for y in range(pix.shape[1])]
    middle = [ pix.shape[2]/3 < x < 2*pix.shape[2]/3 for x in range(pix.shape[2]) ]
    mask = np.einsum('j,k->jk', lower_half, middle)
    bone2 = np.einsum('ijk,jk->ijk', bone, mask)
    bone3 = binary_opening(bone2, iterations=1)
    density_z = np.sum(bone3, axis=(1,2))
    '''
    density_z = np.sum(bone_mask[2*pix.shape[0]/5:4*pix.shape[0]/5, :, :], axis=(1,2))
    
    f, Y = periodogram(density_z, fs=1., detrend='linear')
    fcrit = 1./25
    indices = np.where(f > fcrit)
    f = f[indices]
    Y = Y[indices]
    peaks = find_peaks_cwt(Y, widths=np.arange(0.1,0.3))
    position, maxpeak = max(enumerate(Y[peaks]), key=lambda e: e[1]) # choose max peak
    n_layers_mode = 1./f[peaks[position]]
    n_layers_avg = np.sum( Y[peaks] / f[peaks] ) / np.sum(Y[peaks]) # weighted average of peaks to find avg distance
    n_layers_std = np.sqrt(__get_weighted_var__(peaks, Y, f, n_layers_avg))
    n_perc_mode = n_layers_mode * 100. / lung_height
    n_perc_avg = n_layers_avg * 100. / lung_height
    n_perc_std = n_layers_std * 100. / lung_height

    if DEBUG:
#        cube_show_slider(pix_resampled*mask)
#        cube_show_slider(bone2)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(range(len(density_z)),density_z)
        ax[0].set_xlabel('Layer')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(1./f, Y, 'r')
        ax[1].set_xlabel('#Layers')
        ax[1].set_ylabel('|Y(freq)|')
        plt.show()
        cube_show_slider(bone_mask[2*pix.shape[0]/5:4*pix.shape[0]/5, :, :])
        
    return {'n_perc_avg': n_perc_avg, 'n_perc_mode': n_perc_mode, 'n_perc_std': n_perc_std }
    
def __center_lungs__(lung_mask, shape):
    lung = np.zeros(shape)
    if shape[1] < lung_mask.shape[1]:
        return crop_image(lung_mask, size=shape[1])
    else:
        return extend_image(lung_mask, val=0, size=shape[1])
    ''' 
    cx = shape[2]/2
    cy = shape[1]/2
    lx = lung_mask.shape[2]/2
    ly = lung_mask.shape[1]/2
    lung[:, cy-ly:cy+ly, cx-lx:cx+lx] = lung_mask
    return lung 
    '''
    
def __to_positive__(pix):
    #return 255.*1000./3500. + 255./3500. * pix
    return 1100 + pix

def __segment_chest__(pix, lung_mask):
    erode = 10
    mask = lung_mask.astype(bool)
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=20)
    mask = scipy.ndimage.morphology.binary_closing(np.logical_or(mask, pix <= -1000), iterations=erode)
    zs = np.array(range(pix.shape[0])) 
    zs = np.logical_or( zs < pix.shape[0]/5, zs > 4*pix.shape[0]/5 )
    xs = np.array(range(pix.shape[2]))
    xs = np.logical_or(xs < erode, xs > pix.shape[2]-erode)
    ys = np.array(range(pix.shape[1]))
    ys = ys < pix.shape[1] / 2
    mask_xy = np.einsum('j,k->jk', ys, np.ones(pix.shape[2]))
    mask_z = np.einsum('i,jk->ijk', zs, np.ones(shape=(pix.shape[1], pix.shape[2])))
    mask_x = np.einsum('ij,k->ijk', np.ones(shape=(pix.shape[0], pix.shape[1])), xs)
    mask_y = np.einsum('ik,j->ijk', np.ones(shape=(pix.shape[0], pix.shape[2])), np.array(range(pix.shape[1])) < erode)
    mask = np.einsum('ijk,jk->ijk', mask, mask_xy)
    mask = np.logical_or(mask, mask_z)
    mask = np.logical_or(mask, mask_x)
    mask = np.logical_or(mask, mask_y)
    mask = np.logical_or(mask, np.einsum('i,jk->ijk',1-zs,1-mask_xy))
    labels = np.array([label(img_z) for img_z in 1-mask])
    for img_label in labels:
        areas = [r.area for r in regionprops(img_label)]
        areas.sort()
        if len(areas) >= 1:
            for region in regionprops(img_label):
                if region.area < areas[-1]:
                    for coordinates in region.coords:
                        img_label[coordinates[0], coordinates[1]] = 0
    return labels > 0
    
def __get_sex_predictors__(pix, mask):
    masked = np.ma.array(pix, mask=(1-mask).astype(bool), fill_value=0)
    mu = np.ma.mean(masked)
    sd = np.ma.std(masked)
    img = (masked - mu) / sd
    imgarr = np.ma.filled(img)
    #edges = np.array([ filters.frangi(imgarr[z,:,:], beta1=0.01, beta2=1.0) for z in xrange(imgarr.shape[0]) ])
    edges_re, edges_im = map(np.array, zip(*[ filters.gabor(imgarr[z,:,:], frequency=10) for z in xrange(imgarr.shape[0]) ]))
    binary = (edges_re > 10).astype(int)
    
    i = 0
    cube_show_slider(img)
    while i < 2:
        binary = scipy.ndimage.morphology.binary_opening(binary, iterations=1).astype(int)
        
        
        labels = label(binary, neighbors=8)
        for img_label in labels:
            areas = [r.area for r in regionprops(img_label)]
            areas.sort()
            if len(areas) >= 1:
                for region in regionprops(img_label):
                    if region.area == areas[-1] or region.area > 0.25*areas[-1] or region.area < 25:
                        for coordinates in region.coords:
                            img_label[coordinates[0], coordinates[1]] = 0
        binary = labels > 0
        binary = np.array([scipy.ndimage.morphology.binary_closing(bin, iterations=2).astype(int) for bin in binary])
        print np.bincount(labels.flatten())
        print np.sum(labels > 0)
        i += 1
    

        cube_show_slider(binary, vmin=0, vmax=1)
        cube_show_slider(labels, vmin=0, vmax=1)
#    cube_show_slider(edges_re, vmin=np.min(edges_re), vmax=np.max(edges_re))
#    cube_show_slider(edges_im, vmin=np.min(edges_im), vmax=np.max(edges_im))
    return {'volume_pix': np.sum(labels > 0)}
    '''
    moms = [moments(img_z) for img_z in img]
    moms_central = [moments_central(img[z,:,:], moms[z][0, 1] / moms[z][0, 0], moms[z][1, 0] / moms[z][0, 0]) for z in xrange(img.shape[0])]
    moms_central_norm = [ moments_normalized(mom_central) for mom_central in moms_central ]
    moms_hu = np.array([moments_hu(mom_central_norm) for mom_central_norm in moms_central_norm])
    means = np.mean(moms_hu, axis=0)
    
    features = dict({'below0':np.sum(img < 0), 'totalmask': np.sum(mask), 'fraction': np.sum(img<0) * 1. / np.sum(mask)})
    return dict( dict(zip(['mean_moments_hu_'+str(i) for i in xrange(7)], means)), **features)
    '''
    '''
    imgprops = [regionprops(bin, cache=True) for bin in binary if np.any(bin > 0)]
    features = dict({'area': np.mean(np.array([prop[0].area for prop in imgprops])), \
                     'eccentricity': np.mean(np.array([prop[0].eccentricity for prop in imgprops])), \
                     'extent': np.mean(np.array([prop[0].extent for prop in imgprops])), \
                     'perimeter': np.mean(np.array([prop[0].perimeter for prop in imgprops])), \
                     'solidity': np.mean(np.array([prop[0].solidity for prop in imgprops])), \
                     })
    momentsdict = dict(zip([ 'mean_moments_hu_'+str(i) for i in xrange(7) ], np.mean(np.array([prop[0].moments_hu for prop in imgprops]), axis=0)))  
    '''
    
    #print dict(features, **momentsdict)
    #return dict(features, **momentsdict)
    
'''    
def __get_sex_predictors__(pix, mask):
    img = np.array([img_z for img_z in __to_positive__(pix)*(1-mask) if np.any(img_z)]).astype(float)
    #cube_show_slider(__3d_sobel__(pix)*(1-mask) > 300, vmin=0, vmax=1)
    previous_thing = np.abs(__3d_sobel__((1-mask)*pix))
    thing = scipy.ndimage.morphology.binary_dilation(np.logical_and(previous_thing < 600, previous_thing > 200), iterations=1)
    #cube_show_slider(1-scipy.ndimage.morphology.binary_dilation(mask, iterations=1).astype(float), vmin=0.0, vmax=1.0)
    cube_show_slider(previous_thing.astype(float), vmin=0, vmax=1)
    #cube_show_slider(img)
    moms = [moments(img_z, order=6) for img_z in img]
    moms_central = [moments_central(img[z,:,:], moms[z][0, 1] / moms[z][0, 0], moms[z][1, 0] / moms[z][0, 0], order=6) for z in xrange(img.shape[0])]
    moms_central_norm = [ moments_normalized(mom_central, order=6) for mom_central in moms_central ]
    moms_hu = np.array([moments_hu(mom_central_norm) for mom_central_norm in moms_central_norm])
    #minvals = np.min(moms_hu, axis=0)
    #print minvals
    #summary = describe(np.log10(moms_hu + 1.0 - minvals), axis=0)
    #summary = describe(moms_hu, axis=0)
    
    nobs, minmax, mean, variance, skewness, kurtosis = describe(moms_hu, axis=0)
    return dict(zip([ descr + '_moments_hu_'+str(i) for descr in ['mean','variance','skewness','kurtosis'] for i in xrange(7) ], np.concatenate((mean,variance,skewness,kurtosis))))
    #return dict(zip(['moment_'+str(i) for i in xrange(7)], moms_hu))
'''
        
def process_patient_file(patient_file):
    print patient_file.split('/')[-1]
    patient = load_scan(os.path.join(path, patient_file))
    pix = get_pixels_hu(patient)  # From pixels to HU
#    pix = np.flip(pix, axis=0)
    pix[pix < -1500] = -3400
    original_spacing = dicom_get_spacing(patient)
    pix_resampled, new_spacing = resample(pix, spacing=original_spacing, new_spacing=common_spacing)
    
    #cube_show_slider(pix_resampled)
    
    features = {}
        
    preprocessed_pix = np.load(os.path.join(preprocessed, 'dsb_'+patient_file+'.npz'))['arr_0']
    lung_mask = preprocessed_pix[1,:,:,:]
    pix_resampled = resize_image(pix_resampled, size=(pix_resampled.shape[1]-pix_resampled.shape[1]%2))
    lung_mask = __center_lungs__(lung_mask, pix_resampled.shape)
    
    lung_height = __get_lung_height__(lung_mask)

    if lung_height == 0:
        return {}
    
    '''
    bone_mask = segment_bones(pix_resampled)
    features = dict(features, **get_intercostal_dist(pix_resampled, bone_mask, new_spacing, lung_height))
  
#    cube_show_slider(bone_mask)
    bone_vol = np.sum(bone_mask)
    true_bone = np.logical_and(pix_resampled > 350, pix_resampled < 2500)
    bone_mass = np.sum(pix_resampled * bone_mask * true_bone )
    
    features = dict({ 'bone_density': bone_mass * 1. / bone_vol }, **features) 
    '''
    
    chest_mask = __segment_chest__(pix_resampled, lung_mask)
    
    #cube_show_slider(chest_mask, vmin=0, vmax=1)
    sex_predictors = __get_sex_predictors__(pix_resampled, chest_mask)
    #print sex_predictors
    features = dict(sex_predictors, **features) 
    return features

    
if __name__ == "__main__":
    print 'server:', SERVER
    print 'debug:', DEBUG
    
    if SERVER and not DEBUG:
        p = Pool()
        features = p.map(process_patient_file, patient_files)
        p.close()
        p.join()
    elif not DEBUG:
        features = []
        for patient_file in patient_files:
            features.append(process_patient_file(patient_file))
    else:
        features = [] 
        for patient_file in patient_files[:10]:
            features.append(process_patient_file(patient_file))
    
    if not DEBUG:
        with open(output_file,'w') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id'] + features[0].keys())
            writer.writeheader()
            for i, d in zip(patient_files, features):
                writer.writerow(dict({'patient_id': i}, **d))


# Mingot
wp = os.environ['LUNG_PATH']
INPUT_PATH = wp + 'data/preprocessed5_sample'
p = np.load(INPUT_PATH + '/' + patient_files[5])['arr_0']

lung_mask = __center_lungs__(p[1], p[0].shape)
lung_height = __get_lung_height__(lung_mask)
chest_mask = __segment_chest__(p[0], lung_mask)

sex_predictors = __get_sex_predictors__(p[0], chest_mask)

cube_show_slider(p[0])
cube_show_slider(chest_mask)
cube_show_slider(lung_mask)
plt.imshow(lung_mask[85])
plt.imshow(chest_mask[85])
plt.show()