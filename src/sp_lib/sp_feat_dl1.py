import os
import pandas as pd
import numpy as np
import re

from math import sqrt,ceil
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import skimage.feature as skf
import skimage.filters as skfi

from skimage.filters.rank import entropy
from skimage.morphology import disk, square

import matplotlib.pyplot as plt
import pickle as pk
import scipy.misc as spm

# PATH_DSB = "/media/sergi/Seagate Expansion Drive/preprocessed5/"
PATH_DSB = "/home/sergi/all/devel/big/lung_cancer_ds_bowl/preprocessed5/"
PATH_LUNG = os.environ.get('LUNG_PATH')
PATH_SRC = PATH_LUNG + "/src"
os.chdir(PATH_SRC)
PATH_OUT = PATH_LUNG + "/data/sp_data/dl/"
PATH_DL = PATH_LUNG + "/data/output_dl_example.csv"

# load and filter DL results
# removed last row (was half written)
df_dl = pd.read_csv(PATH_DL)
df_dl_filter = df_dl[df_dl['x'].between(1, 510) & 
    df_dl['y'].between(1, 510) & 
    df_dl['diameter'].between(3.5, 28)]

# explore data
df_grouped = df_dl_filter.groupby("filename")
df_grouped.sum()


# k = np.load(PATH_DSB + 'dsb_00edff4f51a893d80dae2d42a7f45ad1.npz')['arr_0']
# k.shape

# a=df_dl.iloc[[1]]
# a['diameter']

# my crappy functions
import sp_lib.model as splm

# Example code for a given patient 
test_pat = 'dsb_00edff4f51a893d80dae2d42a7f45ad1.npz'
#test_pat = 'luna_286647622786041008124419915089.npz'

# (1) Extract patchs from data frame and one patient
p_patch, p_df = splm.extract_patch(
    df_dl_filter, 
    patient_id=test_pat, 
    patient_path=PATH_DSB, 
    patient_colname='filename',
    swap_xy=False, 
    verbose=False)
# Images for this patient: original patch, rescaled patch, 
#   original lung mask, rescaled lung mask
# We will use the rescaled patches (hu units)
# not filtered by HU units at the moment

# (2) Extract properties (not features yet)
p_prop = [splm.process_img(img['resc_hu'], img['resc_lung']) for img in p_patch]
# Extract meaningful features
# TODO: also use (weighted?) hu moments, HOG, LBP, use lung mask in the process
# this returns 1-row dfs for each patch, or None 

# (3) extract properties
p_feat = [splm.process_prop(p) for p in p_prop]

# p_filtered = [x is None for x in p_feat]
# p_all = zip(p_df, p_feat)[p_filtered]
# df_all = pd.concat([p_df.iloc[[ind]].astype(dict).append(feat) for ind, feat in enumerate(p_feat) if feat is not None])

# Combine all data frames
# removed slices 

# (4) indices of the non-null patches (some patches are null because 
# segmentation in (2) did not find anything)
patch_nonnull = [x is not None for x in p_feat]

# data frame with features
# (5) data_frame of features
df_feat = pd.concat(p_feat)
# recover indices
df_feat.index = p_df.index[patch_nonnull]

# (6) concat data frames to obtain the final augmented data frame for this patient
# df_all = pd.merge(p_df.iloc[patch_nonnull], df_feat, how='cross')
df_all = pd.concat([p_df.iloc[patch_nonnull], df_feat], axis=1)
# (p_df.iloc[patch_nonnull]).shape
# df_feat['patientid'] = p_df.patientid[[x is not None for x in p_feat]].tolist()
# df_all = p_df.merge(df_feat, on = 'patientid')

# (7) see it worked
df_all.head()

# plot intermediate images
# p_img = [splm.process_img(img['resc_hu'], img['resc_lung'], return_images=True) for img in p_patch]
# splm.process_plot(p_img[00:10])


# visualise the patient and reported patches
# len(p_patch), len(p_prop), len(p_feat)

# x_pat = np.load(PATH_DSB + test_pat)['arr_0']

# df_dl_filter.iloc[0:10]
# df_dl_filter[df_dl_filter['patientid'] == test_pat].iloc[0:10]
# plt.imshow(x_pat[1, 53])
# from utils.plotting import cube_show_slider

# cube_show_slider(x_pat[0]*x_pat[1])
# plt.imshow(x_pat[0, 47])
