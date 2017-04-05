"""
File for preprocessing the datasets. It gets as input the DCOMS path, and converts it to numpy.


Datasets accepted: ['dsb', 'lidc', 'luna']

Example usage:
python 00_preprocess.py --input=/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/subset0 --output=/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preproc_luna --pipeline=luna --nodules=/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/annotations.csv
python 00_preprocess.py --input=/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/sample_images --output=/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preproc_dsb --pipeline=dsb

python 00_preprocess.py --input=/home/shared/data/luna/images --output=/mnt/hd2/preprocessed5/ --pipeline=luna --nodules=/home/shared/data/luna/annotations.csv
python 00_preprocess.py --input=/home/shared/data/stage1 --output=/mnt/hd2/preprocessed5/ --pipeline=dsb
"""

import os
import sys
from glob import glob
from time import time
import SimpleITK as sitk
import numpy as np
import pandas as pd
from utils import preprocessing
from utils import reading
from utils import lung_segmentation

import matplotlib.pyplot as plt
from utils import plotting


# Define folder locations
wp = os.environ.get('LUNG_PATH', '')
INPUT_FOLDER = os.path.join(wp, 'data/luna/luna0123')
OUTPUT_FOLDER = os.path.join(wp, 'data/stage1_proc/')
NODULES_PATH = os.path.join(wp, 'data/luna/annotations.csv')
PIPELINE = 'dsb'  # for filename

# Define parametres
COMMON_SPACING = [2, 0.7, 0.7]

# Execution parameters
SAVE_RESULTS = True

# Overwriting parameters by console
for arg in sys.argv[1:]:
    if arg.startswith('--input='):
        INPUT_FOLDER = ''.join(arg.split('=')[1:])
    elif arg.startswith('--output='):
        OUTPUT_FOLDER = ''.join(arg.split('=')[1:])
    elif arg.startswith('--pipeline='):
        PIPELINE = ''.join(arg.split('=')[1:])
    elif arg.startswith('--nodules='):
        NODULES_PATH = ''.join(arg.split('=')[1:])
    else:
        print('Unknown argument {}. Ignoring.'.format(arg))


if PIPELINE == 'dsb':
    patient_files = os.listdir(INPUT_FOLDER)
elif PIPELINE == 'luna':
    patient_files = glob(INPUT_FOLDER + '/*.mhd')  # patients from subset
    df_nodules = pd.read_csv(NODULES_PATH)

## get IDS in the output folder to avoid recalculating them
current_ids = glob(OUTPUT_FOLDER + '/*.npz')
current_ids = [x.split('_')[-1].replace('.npz', '') for x in current_ids]


# Main loop over the ensemble of the database
times = []
for patient_file in patient_files:
    
    tstart = time()
    nodule_mask = None
    print('Processing patient: %s' % patient_file)
    
    # Read
    try:
        if PIPELINE == 'dsb':
            patient = reading.load_scan(os.path.join(INPUT_FOLDER, patient_file))
            patient_pixels = preprocessing.get_pixels_hu(patient)  # From pixels to HU
            originalSpacing = reading.dicom_get_spacing(patient)
            pat_id = patient_file

        elif PIPELINE == 'luna':
            patient = sitk.ReadImage(patient_file) 
            patient_pixels = sitk.GetArrayFromImage(patient)  # indexes are z,y,x
            originalSpacing = [patient.GetSpacing()[2], patient.GetSpacing()[0], patient.GetSpacing()[1]]
            pat_id = patient_file.split('.')[-2]

            # load nodules
            seriesuid = patient_file.split('/')[-1].replace('.mhd', '')
            nodules = df_nodules[df_nodules["seriesuid"] == seriesuid]  # filter nodules for patient
            nodule_mask = reading.create_mask(img=patient, nodules=nodules)

            if nodule_mask is None:  # create virtual nodule mask for coherence
                nodule_mask = np.zeros(patient_pixels.shape, dtype=np.int)

    except Exception as e:  # Some patients have no data, ignore them
        print('There was some problem reading patient {}. Ignoring and live goes on.'.format(patient_file))
        print('Exception', e)
        continue

    # avoid computing the id if not already present
    if pat_id in current_ids:
        continue

    # SET BACKGROUND: set to air parts that fell outside
    patient_pixels[patient_pixels < -1500] = -2000


    # RESAMPLING
    pix_resampled, new_spacing = preprocessing.resample(patient_pixels, spacing=originalSpacing, new_spacing=COMMON_SPACING)
    if nodule_mask is not None:
        nodule_mask, new_spacing = preprocessing.resample(nodule_mask, spacing=originalSpacing, new_spacing=COMMON_SPACING)
    print('Resampled image size: {}'.format(pix_resampled.shape))


    # LUNG SEGMENTATION (if TH1 fails, choose TH2)
    lung_mask = lung_segmentation.segment_lungs(image=pix_resampled, fill_lung=True, method="thresholding1")
    if not lung_segmentation.is_lung_segmentation_correct(lung_mask):
        lung_mask = lung_segmentation.segment_lungs(image=pix_resampled, fill_lung=True, method="thresholding2")


    # CROPPING to 512x512
    pix = preprocessing.resize_image(pix_resampled, size=512)  # if zero_centered: -0.25
    lung_mask = preprocessing.resize_image(lung_mask, size=512)
    if nodule_mask is not None:
        nodule_mask = preprocessing.resize_image(nodule_mask, size=512)
    print('Cropped image size: {}'.format(pix.shape))


    # STACK results
    output = np.stack((pix, lung_mask, nodule_mask)) if nodule_mask is not None else np.stack((pix, lung_mask))


    if SAVE_RESULTS:
        np.savez_compressed(os.path.join(OUTPUT_FOLDER, "%s_%s.npz") % (PIPELINE, pat_id), output)

    x = time()-tstart
    print('Patient {}, Time: {}'.format(pat_id, x))
    times.append(x)

print('Average time per image: {}'.format(np.mean(times)))

