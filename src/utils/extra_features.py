import os, csv
import numpy as np
import SimpleITK as sitk
from glob import glob
from reading import *
from plotting import *
from preprocessing import *
from segmentation import *
from scipy.ndimage.morphology import *
from scipy.signal import periodogram, find_peaks_cwt
from multiprocessing import Pool

import matplotlib.pyplot as plt

DEBUG = False
SERVER = os.uname()[1] == 'ip-172-31-7-211'

if SERVER:
    path = '/home/shared/data/stage1'
    preprocessed = '/mnt/hd2/preprocessed4'
    output_file = '/home/shared/data/stage1_extra_features.csv'
else:
    path = '/home/carlos/DSB2017/dsb_sample'
    output_file = '/home/carlos/lung_cancer_ds_bowl/data/stage1_extra_features.csv'

patient_files = os.listdir(path)
patient_files = sorted(patient_files)

#patient_files = ['6540e089936a6c9d1f11939e96e6ab18']

common_spacing = [2., 0.6, 0.6]

'''
def segment_bones(image):
    bone = np.logical_and(image > 350, image < 2500)
    bone = scipy.ndimage.morphology.binary_dilation(bone, iterations=2)
    cube_show_slider(bone)
    plt.imshow(sobel(bone[50,:,:]))
    plt.show()
    return 
'''

def __get_weighted_var__(indices, y, f, avg):
    n = len(np.where(y[indices] != 0)[0])
    if n > 1:
        return n * 1. / (n-1) * np.sum(y[indices]*(1./f[indices] - avg)**2) / np.sum(y[indices])
    else:
        return 0.

def __get_lung_height__(lung_mask):
    exists_mask = np.where(np.sum(lung_mask, axis=(1,2)) > 0)
    return np.max(exists_mask) - np.min(exists_mask)
    

def get_intercostal_dist(pix, new_spacing, lung_height):
    bone = np.logical_and(pix > 350, pix < 2500)
    lower_half = [y > pix.shape[1]/2 for y in range(pix.shape[1])]
    middle = [ pix.shape[2]/3 < x < 2*pix.shape[2]/3 for x in range(pix.shape[2]) ]
    mask = np.einsum('j,k->jk', lower_half, middle)
    bone2 = np.einsum('ijk,jk->ijk', bone, mask)
    density_z = np.sum(binary_opening(bone2, iterations=1), axis=(1,2))
    
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
        cube_show_slider(pix_resampled*mask)
        cube_show_slider(bone2)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(range(len(density_z)),density_z)
        ax[0].set_xlabel('Layer')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(1./f, Y, 'r')
        ax[1].set_xlabel('#Layers')
        ax[1].set_ylabel('|Y(freq)|')
        plt.show()
        
    return {'n_perc_avg': n_perc_avg, 'n_perc_mode': n_perc_mode, 'n_perc_std': n_perc_std }
    
def process_patient_file(patient_file):
    print patient_file.split('/')[-1]
    patient = load_scan(os.path.join(path, patient_file))
    pix = get_pixels_hu(patient)  # From pixels to HU
    pix = np.flip(pix, axis=0)
    pix[pix < -1500] = -3400
    original_spacing = dicom_get_spacing(patient)
    pix_resampled, new_spacing = resample(pix, spacing=original_spacing, new_spacing=common_spacing)
    
    features = {}
    
    if SERVER:
        preprocessed_pix = np.load(os.path.join(preprocessed, patient_file))['arr_0']
        lung_mask = preprocessed_pix[1,:,:,:]
    else:
        lung_mask = segment_lungs(pix_resampled)
    
    lung_height = __get_lung_height__(lung_mask)
    
    features = dict(features, **get_intercostal_dist(pix_resampled, new_spacing, lung_height))

#    bone_mask = segment_bones(pix_resampled)
#    cube_show_slider(bone_mask)
#    return
#    bone_vol = np.sum(bone_mask)
    return features    
    

    
if __name__ == "__main__":
    print 'server:', SERVER
    print 'debug:', DEBUG
    p = Pool()
    features = p.map(process_patient_file, patient_files[:5])
#    for patient_file in patient_files:
#        print 'ID:', patient_file
#        process_patient_file(patient_file)
    p.close()
    p.join()
    if not DEBUG:
        with open(output_file,'w') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id'] + features[0].keys())
            writer.writeheader()
            for i, d in zip(patient_files, features):
                writer.writerow(dict({'patient_id': i}, **d))
