import os, csv
import numpy as np
import SimpleITK as sitk
from glob import glob
from reading import *
from plotting import *
from preprocessing import *
from scipy.ndimage.morphology import *
from scipy.signal import periodogram, find_peaks_cwt
from multiprocessing import Pool

import matplotlib.pyplot as plt

path = '/home/shared/data/stage1'
output_file = '/home/shared/data/stage1_extra_features.csv'

#path = '/home/carlos/DSB2017/dsb_sample'
#output_file = '/home/carlos/lung_cancer_ds_bowl/data/stage1_extra_features_mean.csv'

patient_files = os.listdir(path)
patient_files = sorted(patient_files)

#patient_files = ['6540e089936a6c9d1f11939e96e6ab18']

common_spacing = [2., 0.6, 0.6]

def process_patient_file(patient_file):
#    print patient_file.split('/')[-1][:-4]
    patient = load_scan(os.path.join(path, patient_file))
    pix = get_pixels_hu(patient)  # From pixels to HU
    pix = np.flip(pix, axis=0)
    pix[pix < -1500] = -3400
    original_spacing = dicom_get_spacing(patient)
    pix_resampled, new_spacing = resample(pix, spacing=original_spacing, new_spacing=common_spacing)
    
    bone = np.logical_and(pix_resampled > 350, pix_resampled < 2500)
    lower_half = [y > pix_resampled.shape[1]/2 for y in range(pix_resampled.shape[1])]
    middle = [ pix_resampled.shape[2]/3 < x < 2*pix_resampled.shape[2]/3 for x in range(pix_resampled.shape[2]) ]
    mask = np.einsum('j,k->jk', lower_half, middle)
    bone2 = np.einsum('ijk,jk->ijk', bone, mask)
    density_z = np.sum(binary_opening(bone2, iterations=1), axis=(1,2))
    
    f, Y = periodogram(density_z, fs=1., detrend='linear')
    fcrit = 1./25
    indices = np.where(f > fcrit)
    f = f[indices]
    Y = Y[indices]
    peaks = find_peaks_cwt(Y, widths=np.arange(0.1,0.3))
    n_layers = np.sum( Y[peaks] / f[peaks] ) / np.sum(Y[peaks])
#    position, maxpeak = max(enumerate(Y[peaks]), key=lambda e: e[1])
#    n_layers = 1./f[peaks[position]]
    n_mm = n_layers * new_spacing[0]
    
#    print n_layers
#	print 1./f[peaks], Y[peaks], maxpeak
    '''    
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
    '''
    return n_mm
    
if __name__ == "__main__":
	p = Pool()
	distances = p.map(process_patient_file, patient_files)
	p.close()
	p.join()
	
	with open(output_file,'w') as f:
		writer = csv.DictWriter(f, fieldnames=['patient_id', 'intercostal_dist_mm'])
		
		writer.writeheader()
		for i, d in zip(patient_files, distances):
		    writer.writerow({'patient_id': i, 'intercostal_dist_mm': d})
	
