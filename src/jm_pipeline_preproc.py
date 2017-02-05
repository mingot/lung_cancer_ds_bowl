"""
File for preprocessing the datasets. It gets as input the DCOMS path, and converts it to numpy.


Datasets accepted:
    -DSB
"""

import os
import numpy as np
from time import time
from utils import reading
from utils import preprocessing
from utils import plotting
import matplotlib.pyplot as plt

# Reading patients
wp = os.environ['LUNG_PATH']
TMP_FOLDER = os.path.join(wp, 'data/jm_tmp/')
INPUT_FOLDER = os.path.join(wp, 'data/sample_images/')  # 'data/stage1/stage1/'
OUTPUT_FOLDER = os.path.join(wp, 'data/stage1_proc/')
PIPELINE = 'dsb'  # for filename

#Overwriting parameters by console
for arg in sys.argv[1:]:
    if arg.startswith('--input='):
        INPUT_FOLDER = ''.join(arg.split('=')[1:])
    elif arg.startswith('--output='):
        OUTPUT_FOLDER = ''.join(arg.split('=')[1:])
    elif arg.startswith('--tmp='):
        TMP_FOLDER =''.join(arg.split('=')[1:])
    elif arg.statswith('--pipeline='):
        PIPELINE = ''.join(arg.split('=')[1:])
    else:
        print 'Unknown argument %s. Ignoring.' %arg
        
patients = os.listdir(INPUT_FOLDER)
patients.sort()

times = []

for patid in patients:
    
    n = time()
    
    # Read
    # patid = patients[10]
    if PIPELINE == 'dsb':
        patient = reading.load_scan(INPUT_FOLDER + patid)
    elif PIPELINE == 'lidc':
        patient = reading.read_patient_lidc(INPUT_FOLDER + patid)
    
    # From pixels to HU
    patient_pixels = preprocessing.get_pixels_hu(patient)
    #patient_pixels.shape
    #plt.imshow(patient_pixels[3])
    #plt.hist(patient_pixels.flatten(), bins=80, color='c')
    
    # Resampling
    pix_resampled, spacing = preprocessing.resample(patient_pixels, patient, [1,1,1])  # slow
    #print("Shape before resampling\t", patient_pixels.shape)
    #print("Shape after resampling\t", pix_resampled.shape)
    #plt.imshow(pix_resampled[50])
    #plotting.plot_3d(pix_resampled, 400)
        
    # Segment lungs
    lung_mask = preprocessing.segment_lung_mask(pix_resampled, fill_lung_structures=True)  # TODO: expand the mask (as suggested in the kernel)
    #lung_mask.shape
    #plt.imshow(lung_mask[90])
    #plotting.plot_3d(segmented_lungs_fill, 0)
    #plotting.plot_3d(segmented_lungs_fill - segmented_lungs, 0)
    
    #Compute volume for sanity test
    lung_volume_l = np.sum(lung_mask)/(100.**3)
    if lung_volume_l < 2 or lung_volume_l > 10:
        print 'Warning lung volume: %s out of physiological values. Double Check segmentation.' % patid 
    
    # zero center and normalization
    pix = preprocessing.normalize(pix_resampled)
    pix = preprocessing.zero_center(pix)
    
    # store output (processed lung and lung mask)
    output = np.stack((pix, lung_mask))
    np.savez_compressed(OUTPUT_FOLDER + "%s_%s.npz" % (PIPELINE, patid), output)  # 10x compression over np.save (~400Mb vs 40Mg), but 10x slower  (~1.5s vs ~15s)
    #np.save("prova.npy", output)
    
    x = time()-n
    print 'Time: %s' % str(x)
    times.append(x)


print 'Average time per image: %s' % str(np.mean(times))
