import os
import numpy as np
from time import time
from utils import reading
from utils import preprocessing
from utils import plotting
import matplotlib.pyplot as plt

# Define folder locations
wp = os.environ['LUNG_PATH']
TMP_FOLDER = wp + 'data/jm_tmp/'
INPUT_FOLDER = wp + 'data/stage1/'
# INPUT_FOLDER = wp + 'data/sample_images/'
OUTPUT_FOLDER = wp + 'data/stage1_proc/'
PIPELINE = 'dsb'  # for filename

# Execution parameters
show_intermediate_images = True

# Small code to get rid of .DS_Store
patients = []
for f in os.listdir(INPUT_FOLDER):
    if not f.startswith('.'):
        patients.append(f)

patients.sort()

# Main loop over the ensemble of teh database
times = []
for pat_id in patients:
    
    n = time()
    
    # Read the input image
    patient = reading.load_scan(INPUT_FOLDER + pat_id)
    
    # From pixels to HU
    patient_pixels = preprocessing.get_pixels_hu(patient)
    if show_intermediate_images:
        print("Shape of raw data\t", patient_pixels.shape)
        plt.figure()
        plt.imshow(patient_pixels[3])
        plt.title('Raw pixel data')
        plt.figure()
        plt.hist(patient_pixels.flatten(), bins=80, color='c')
        plt.title('Histogram')

    # Resampling
    # TODO: Accelerate the resampling
    pix_resampled, spacing = preprocessing.resample(patient_pixels, patient, [1, 1, 1])
    if show_intermediate_images:
        print("Shape after resampling\t", pix_resampled.shape)
        plt.figure()
        plt.imshow(pix_resampled[50])
        plt.title('Resampled data')
        # plt.figure()
        # plotting.plot_3d(pix_resampled, 400)
    
    # Segment lungs
    lung_mask = preprocessing.segment_lung_mask(pix_resampled, fill_lung_structures=True)
    # TODO: expand the mask (as suggested in the kernel)
    if show_intermediate_images:
        lung_mask.shape
        plt.figure()
        plt.imshow(lung_mask[90])
        # plotting.plot_3d(segmented_lungs_fill, 0)
        # plotting.plot_3d(segmented_lungs_fill - segmented_lungs, 0)
    
    # zero center and normalization
    pix = preprocessing.normalize(pix_resampled)
    pix = preprocessing.zero_center(pix)
    
    # store output (processed lung and lung mask)
    output = np.stack((pix, lung_mask))
    # TODO: The following line crashes if the output folder does not exist
    np.savez_compressed(OUTPUT_FOLDER + "%s_%s.npz" % (PIPELINE, pat_id), output)  # 10x compression over np.save (~400Mb vs 40Mg), but 10x slower  (~1.5s vs ~15s)
    # np.save("prova.npy", output)
    
    x = time()-n
    print 'Time: %s' % str(x)
    times.append(x)

print 'Average time per image: %s' % str(np.mean(times))


