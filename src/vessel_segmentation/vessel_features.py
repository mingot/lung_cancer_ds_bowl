import os
import sys
import random
import io
import requests
import pandas as pd
import numpy as np
import dicom
import matplotlib.pyplot as plt
sys.path.append('/home/marti/lung_cancer_ds_bowl/src')
from utils import plotting, reading, preprocessing
from dl_model_patches import common

pd.set_option('precision', 2)
pd.set_option('display.width', 500)

INPUT_FILE = '/home/marti/lung_cancer_ds_bowl/mergedNodules.csv'

df_input = pd.read_csv(INPUT_FILE)

df_input['patientid'] = df_input.apply(lambda x: x['patientid'].split('/')[-1], axis=1)
df_input.sort_values(by=['patientid', 'nslice'], inplace=True)
print(df_input)
#lala = df_input[df_input.patientid == 'dsb_fc545aa2f58509dc6d81ef02130b6906.npz']
#print(lala)
small_df = df_input[df_input.nslicesSpread > 2]

def x_y_min_max(val, pixels_diameter, x_y_margin, max_value=512):
    min_val = max(0, val - int(pixels_diameter/2) -1 - x_y_margin)
    max_val = min(max_value, val + int(pixels_diameter/2) + 1 + x_y_margin)
    return int(min_val), int(max_val)

def z_min_max(val, nslices_spread, z_margin, patient_nslices):
    min_val = max(0, val - nslices_spread // 2 - 1 - z_margin)
    max_val = min(patient_nslices, val + nslices_spread // 2 + 1 + z_margin)
    return int(min_val), int(max_val)

def mount_3D_patch(data, patient_nslices, nslice, nslices_spread, x, y, diameter, x_y_margin, z_margin):
    pixels_diameter = diameter / 0.7
    min_x, max_x = x_y_min_max(x, pixels_diameter, x_y_margin)
    min_y, max_y = x_y_min_max(y, pixels_diameter, x_y_margin)
    min_z, max_z = z_min_max(nslice, nslices_spread, z_margin, patient_nslices)
    return data[0, min_z:max_z, min_y:max_y, min_x:max_x]

def print_patches(data, nslice, nslices_spread, x, y, diameter):
    patient_nslices = data.shape[1]
    print "patient_nslices: ", patient_nslices
    patch_3D = mount_3D_patch(data=data, patient_nslices=patient_nslices, nslice=nslice, nslices_spread=nslices_spread,
                              x=x, y=y, diameter=diameter, x_y_margin=10, z_margin=3)
    return patch_3D


i = 0
last_patient = None
total_rows = small_df.shape[0]
for row_id, row in small_df.iterrows():
    i += 1
    # if i > 3:
    #    break
    if row['patientid'] != last_patient:
        last_patient = row['patientid']
        # print("Patient " + last_patient)
        patidf = "/mnt/hd2/preprocessed5/%s" % last_patient
        pdata = np.load(patidf)['arr_0']
    # print(row_id)
    x, y, diameter, nslices_spread, nslice = row['x'], row['y'], row['diameter'], row['nslicesSpread'], int(
        row['nslice'])
    # a = common.AuxRegion(bbox=[x-20,y-20,x+20,y+20])
    # plotting.plot_bb(pdata[0,nslice], [a])

    patch = print_patches(data=pdata, nslice=nslice, nslices_spread=nslices_spread, x=y, y=x, diameter=diameter)

    # print("getting the vessel mask...")
    from vessel_segmentation.vessel_segmentation import get_vessel_mask

    msk.type = np.float
    msk, bin_mask = get_vessel_mask(patch)
    small_df.loc[row_id, 'vessel_sum'] = msk.sum()
    small_df.loc[row_id, 'vessel_mean'] = msk.mean()
    small_df.loc[row_id, 'vessel_max'] = msk.max()
    small_df.loc[row_id, 'vessel_min'] = msk.min()

    if i % 50 == 0:
        print(str(i) + '/' + str(total_rows))

        # from utils import plotting
        # plotting.multiplot(patch)
        # plotting.multiplot(msk)
        # plotting.multiplot(bin_mask)

small_df.to_csv('vessel_features.csv')