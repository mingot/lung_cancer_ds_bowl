
import numpy as np
import os
from src.utils import plotting
from src.utils import reading
from src.utils import preprocessing
import random
from glob import glob
import matplotlib.pyplot as plt

import SimpleITK as sitk

## Loading
# wp = os.environ['LUNG_PATH']
# OUTPUT_FOLDER  = wp + 'data/preproc_luna/*.npz'  # Test DSB
# OUTPUT_FOLDER  = wp + 'data/preproc_dsb/*.npz'  # Test luna
# OUTPUT_FOLDER = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preproc_dsb/*.npz'
# OUTPUT_FOLDER = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preproc_luna/*.npz'


# carga luna orginal
OF = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/subset0/*.mhd'
patient_files = glob(OF)
patient = sitk.ReadImage(patient_files[20])
patient = sitk.GetArrayFromImage(patient) #indexes are z,y,x
plt.imshow(patient[40])
plt.show()

x = []
z = []
for p in patient_files:
    patient = sitk.ReadImage(p)
    x.append(patient.GetSpacing()[1])
    z.append(patient.GetSpacing()[2])
    print str(patient.GetSpacing()[0]) + ',' + str(patient.GetSpacing()[1]) + ',' + str(patient.GetSpacing()[2])

plt.hist(x, bins=80)
plt.show()

plt.hist(z, bins=80)
plt.show()

from skimage import draw

img = np.zeros((100, 100), dtype=np.uint8)
rr, cc = draw.circle(50,50,50)
img[rr,cc] = 1

plt.imshow(img)
plt.show()


## carga DSB original
OF = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/sample_images'
patient_files = os.listdir(OF)
patient = preprocessing.get_pixels_hu(reading.load_scan(os.path.join(OF, patient_files[0])))
plt.imshow(patient[50])
plt.show()

plt.hist(patient[40].flatten(), bins=80)
plt.show()

x = []
z = []
for p in patient_files:
    px = reading.load_scan(os.path.join(OF, p))
    spacing = reading.dicom_get_spacing(px)
    x.append(spacing[1])
    z.append(spacing[0])


plt.hist(x, bins=80)
plt.show()

plt.hist(z, bins=80)
plt.show()


# carga npz
OUTPUT_FOLDER = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preproc_dsb/*.npz'
OUTPUT_FOLDER = '/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preproc_luna/*.npz'
proc_file = glob(OUTPUT_FOLDER)  # patients from subset1
pfile = random.choice(proc_file)
p = np.load(proc_file[4])['arr_0']  # abrir archivo
p.shape


# 3d sliding plot
plotting.cube_show_slider(p[0])  # image
plotting.cube_show_slider(p[1])  # lung mask
plotting.cube_show_slider(p[2])  # nodule mask (if exists)


# histogram
plt.hist(p[0,100].flatten(), bins=80)
plt.show()

# study nodules mask
for i in range(p[2].shape[0]):
    if np.max(p[2,i])!=0:
        print i

# plt.imshow(p[2,43])
# plt.show()

plotting.plot_mask(p[0,58], p[2,58])

