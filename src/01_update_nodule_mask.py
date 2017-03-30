import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import multiprocessing
import logging
from utils import reading, preprocessing

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

annotations_df = pd.read_csv("/home/shared/data/luna/annotations.csv")
seriesuids = list(set(annotations_df['seriesuid']))
logging.info("Total patients: %d" % len(seriesuids))
COMMON_SPACING = [2, 0.7, 0.7]

patients_done = os.listdir('/mnt/hd2/aux_validation/') + os.listdir('/mnt/hd2/aux/') + ['luna_225515255547637437801620523312.npz']
seriesuids = [s for s in seriesuids if 'luna_%s.npz' % s.split('.')[-1] not in patients_done]
logging.info("Total patients (undone): %d" % len(seriesuids))


def update_patient(seriesuid):
    logging.info("Loading patient %s" % seriesuid)
    patient = sitk.ReadImage('/home/shared/data/luna/images/%s.mhd' % seriesuid)
    originalSpacing = [patient.GetSpacing()[2], patient.GetSpacing()[0], patient.GetSpacing()[1]]
    # lung_image = sitk.GetArrayFromImage(patient)  # indexes are z,y,x
    nodules = annotations_df[annotations_df["seriesuid"] == seriesuid]  # filter nodules for patient

    nodule_mask = reading.create_mask(img=patient, nodules=nodules)
    nodule_mask, new_spacing = preprocessing.resample(nodule_mask, spacing=originalSpacing, new_spacing=COMMON_SPACING)
    nodule_mask = preprocessing.resize_image(nodule_mask, size=512)

    patid = seriesuid.split('.')[-1]

    # update
    try:
        pat_data = np.load('/mnt/hd2/preprocessed5/luna_%s.npz' % patid)['arr_0']
        pat_data[2] = nodule_mask
        np.savez_compressed("/mnt/hd2/aux/luna_%s.npz" % patid, pat_data)
    except:
        pat_data = np.load('/mnt/hd2/preprocessed5_validation_luna/luna_%s.npz' % patid)['arr_0']
        pat_data[2] = nodule_mask
        np.savez_compressed("/mnt/hd2/aux_validation/luna_%s.npz" % patid, pat_data)



pool = multiprocessing.Pool(4)
pool.map(update_patient, seriesuids)
