
import os
import numpy as np
import pandas as pd
from utils import preprocessing, lung_segmentation, reading
from dl_model_patches import common

INPUT_FOLDER = ''
DL1_MODEL = ''
COMMON_SPACING = [2, 0.7, 0.7]


## PREPROCESSING ------------------------------------------------------------------------------------------------
patient_file = 'xxxxx'
patient = reading.load_scan(os.path.join(INPUT_FOLDER, patient_file))
patient_pixels = preprocessing.get_pixels_hu(patient)  # From pixels to HU
originalSpacing = reading.dicom_get_spacing(patient)
pat_id = patient_file


# SET BACKGROUND: set to air parts that fell outside
patient_pixels[patient_pixels < -1500] = -2000

# RESAMPLING
pix, new_spacing = preprocessing.resample(patient_pixels, spacing=originalSpacing, new_spacing=COMMON_SPACING)


def is_correct(lung_mask):
    pass

# LUNG_MASK
lung_mask = lung_segmentation.segment_lungs(image=pix, fill_lung=True, method="thresholding1")  
if not is_correct(lung_mask):
    lung_mask = lung_segmentation.segment_lungs(image=pix, fill_lung=True, method="thresholding2")  

# CROPPING to 512x512
pix = preprocessing.resize_image(pix, size=512)  # if zero_centered: -0.25
lung_mask = preprocessing.resize_image(lung_mask, size=512)

# STACK RESULTS
patient_data = np.stack((pix, lung_mask))



## RESNET PREDICTION ------------------------------------------------------------------------------------------------

from keras import backend as K
from dl_networks.sample_resnet import ResnetBuilder
from keras.optimizers import Adam

# Model loading inside the listener thread (otherwise keras complains)
K.set_image_dim_ordering('th')

model = ResnetBuilder().build_resnet_50((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', 'fmeasure'])
model.load_weights(DL1_MODEL)


X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, thickness=1)
preds = model.predict(np.asarray(X), verbose=1)


## FINAL PREDICTOR


