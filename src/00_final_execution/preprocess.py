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
import argparse
from glob import glob
from time import time
import logging
import SimpleITK as sitk
import numpy as np
import pandas as pd
from utils import preprocessing, reading, lung_segmentation


# import matplotlib.pyplot as plt
# from utils import plotting

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# Define parametres
COMMON_SPACING = [2, 0.7, 0.7]



def process_filename(patient_file, output_folder, pipeline='dsb', df_nodules=None):

    nodule_mask = None
    logging.info('Processing patient: %s' % patient_file)
    
    # Read
    try:
        if pipeline == 'dsb':
            patient = reading.load_scan(patient_file)
            patient_pixels = preprocessing.get_pixels_hu(patient)  # From pixels to HU
            originalSpacing = reading.dicom_get_spacing(patient)
            pat_id = patient_file.split('/')[-1]

        elif pipeline == 'luna':
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
        logging.error('There was some problem reading patient {}. Ignoring and live goes on.'.format(patient_file))
        logging.error('Exception', e)
        return

    # SET BACKGROUND: set to air parts that fell outside
    patient_pixels[patient_pixels < -1500] = -2000


    # RESAMPLING
    pix_resampled, new_spacing = preprocessing.resample(patient_pixels, spacing=originalSpacing, new_spacing=COMMON_SPACING)
    if nodule_mask is not None:
        nodule_mask, new_spacing = preprocessing.resample(nodule_mask, spacing=originalSpacing, new_spacing=COMMON_SPACING)
    logging.info('Resampled image size: {}'.format(pix_resampled.shape))


    # LUNG SEGMENTATION (if TH1 fails, choose TH2)
    lung_mask = lung_segmentation.segment_lungs(image=pix_resampled, fill_lung=True, method="thresholding1")
    if not lung_segmentation.is_lung_segmentation_correct(lung_mask):
        lung_mask = lung_segmentation.segment_lungs(image=pix_resampled, fill_lung=True, method="thresholding2")


    # CROPPING to 512x512
    pix = preprocessing.resize_image(pix_resampled, size=512)  # if zero_centered: -0.25
    lung_mask = preprocessing.resize_image(lung_mask, size=512)
    if nodule_mask is not None:
        nodule_mask = preprocessing.resize_image(nodule_mask, size=512)
    logging.info('Cropped image size: {}'.format(pix.shape))

    # STACK and save results
    output = np.stack((pix, lung_mask, nodule_mask)) if nodule_mask is not None else np.stack((pix, lung_mask))
    np.savez_compressed(os.path.join(output_folder, "%s_%s.npz") % (pipeline, pat_id), output)


import multiprocessing
def preprocess_files(file_list, output_folder, pipeline='dsb'):
    pool = multiprocessing.Pool(4)  # multiprocessing.cpu_count()
    logging.info("Creating preprocessing job for %d files..." % len(file_list))
    tstart = time()

    #fire off workers
    jobs = []
    for filename in file_list:
        job = pool.apply_async(process_filename, (filename, output_folder, pipeline))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    pool.close()
    pool.join()
    logging.info("Finished preprocessing in %.3f s" % (time()-tstart))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess patient files in parallel')
    parser.add_argument('--input_folder', help='input folder')
    parser.add_argument('--output_folder', help='output folder')
    parser.add_argument('--pipeline', default = 'dsb', help='pipeline to be used (dsb or luna)')
    parser.add_argument('--nodules_csv', help='in case of luna pipeline, nodules annotations')
    args = parser.parse_args()

    # # Define folder locations
    # wp = os.environ.get('LUNG_PATH', '')
    # INPUT_FOLDER = os.path.join(wp, 'data/luna/luna0123')
    # OUTPUT_FOLDER = os.path.join(wp, 'data/stage1_proc/')
    # NODULES_PATH = os.path.join(wp, 'data/luna/annotations.csv')
    # PIPELINE = 'dsb'  # for filename

    patient_files = []
    if args.pipeline=='dsb':
        patient_files = [os.path.join(args.input_folder, p) for p in os.listdir(args.input_folder)]
    elif args.pipeline=='luna':
        patient_files = glob(args.input_folder + '/*.mhd')  # patients from subset
        df_nodules = pd.read_csv(args.nodules_csv)


    # ## get IDS in the output folder to avoid recalculating them
    # current_ids = glob(OUTPUT_FOLDER + '/*.npz')
    # current_ids = [x.split('_')[-1].replace('.npz', '') for x in current_ids]
    preprocess_files(file_list=patient_files, output_folder=args.output_folder, pipeline=args.pipeline)