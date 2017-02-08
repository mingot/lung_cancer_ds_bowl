# xx = np.load(wp + 'data/preproc_luna/luna_100684836163890911914061745866.npz')['arr_0']
# xx = np.load(wp + 'data/preproc_dsb/dsb_0a099f2549429d29b32f349e95fb2244.npz')['arr_0']
# plt.imshow(img[80])
# img[80]

# img.shape
# a = np.zeros((img.shape[0],512, 512))
# a.fill(-0.25)
# a.shape
# img.shape
# x = (512 - img.shape[1])/2
# y = (512 - img.shape[2])/2
# a[:, x:x+img.shape[1], y:y+img.shape[2] ] = img
# plt.imshow(a[200])


## CHECKS for differences luna <> dsb
# import SimpleITK as sitk
# luna_patients = glob(wp + 'data/luna/subset1/*.mhd')  # patients from subset1
# img_file = luna_patients[0]
# itk_img = sitk.ReadImage(img_file)
# img_array = sitk.GetArrayFromImage(itk_img) #indexes are z,y,x
# itk_img.GetSpacing()
# patients = reading.load_scan(os.path.join(INPUT_FOLDER, patients[0]))
# patients
# spacing = map(float, ([patients[0].SliceThickness] + patients[0].PixelSpacing))
# spacing


## TEST 1: check mean pixel distribution of luna vs dsb

import os
os.getcwd()

from src.utils import reading
from src.utils import preprocessing
from src.utils import plotting
from glob import glob
import SimpleITK as sitk


wp = os.environ['LUNG_PATH']

# READING DSB
INPUT_FOLDER = os.path.join(wp, 'data/stage1/')  # 'data/stage1/stage1/'
patient_files = os.listdir(INPUT_FOLDER)
patient = reading.load_scan(os.path.join(INPUT_FOLDER, patient_files[0]))
patient_pixels = preprocessing.get_pixels_hu(patient)
plotting.cube_show_slider(patient_pixels)

# READING LUNA
# load nodes
import pandas as pd
df_node = pd.read_csv(wp + 'data/luna/annotations.csv')
INPUT_FOLDER = os.path.join(wp, 'data/luna/subset0')
idx = 40

patient_file = glob(INPUT_FOLDER + '/*.mhd')[idx]  # patients from subset1
patient = sitk.ReadImage(patient_file)
patient_pixels = sitk.GetArrayFromImage(patient) #indexes are z,y,x
#plotting.cube_show_slider(patient_pixels)
seriesuid = patient_file.split('/')[-1].replace('.mhd','')
nodules = df_node[df_node["seriesuid"]==seriesuid]
nodules

node_x = float(nodules['coordX'])
node_y = float(nodules['coordY'])
node_z = float(nodules['coordZ'])

xx = reading.create_mask(patient, nodules, seriesuid)
plotting.cube_show_slider(xx)

slice_id = 72
plot_mask(patient_pixels[slice_id], xx[slice_id])
# 88, 32

center = np.array([node_x, node_y, node_z])  # nodule center
origin = np.array(patient.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
spacing = np.array(patient.GetSpacing())  # spacing of voxels in world coor. (mm)
v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space
v_center


###########
import random
proc_file = glob(wp + 'data/preproc_luna/*.npz')  # patients from subset1
p = np.load(random.choice(proc_file))['arr_0']
p.shape

plotting.cube_show_slider(p[0])
plotting.cube_show_slider(p[1])
plotting.cube_show_slider(p[2])

proc_file
