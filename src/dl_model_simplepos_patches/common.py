import numpy as np
import logging
import multiprocessing
from time import time
from skimage import measure, transform, morphology
from dl_model_patches.common import *


def load_patient_3d_atlas(   patient_data, patient_nodules_df=None, discard_empty_nodules=False,
                             output_rois=False, debug=False, thickness=0):
    """
    Returns images generated for each patient.
     - patient_nodules_df: pd dataframe with at least: x, y, nslice, diameter
     - thickness: number of slices up and down to be taken
    """
    
    X, Y, rois = [], [], []
    total_stats = {}

    # load the slices to swipe
    if patient_nodules_df is not None:
        nslices = list(set(patient_nodules_df['nslice']))
    else:
        nslices = range(patient_data.shape[1])

    # Check if it has nodules annotated
    if patient_data.shape[0]!=3:
        aux = np.zeros((3,patient_data.shape[1], patient_data.shape[2], patient_data.shape[3]))
        aux[0] = patient_data[0]
        aux[1] = patient_data[1]
        patient_data = aux


    for nslice in nslices:
        lung_image, lung_mask, nodules_mask = patient_data[0,nslice,:,:], patient_data[1,nslice,:,:], patient_data[2,nslice,:,:]


        if patient_nodules_df is None:
            # Discard if no nodules
            if nodules_mask.sum() == 0 and discard_empty_nodules:
                continue

            # Discard if bad segmentation
            voxel_volume_l = 2*0.7*0.7/(1000000.0)
            lung_volume_l = np.sum(lung_mask)*voxel_volume_l
            if lung_volume_l < 0.009 or lung_volume_l > 0.1:
                continue  # skip slices with bad lung segmentation

            # Filter ROIs to discard small and connected
            regions_pred = extract_rois_from_lung_mask(lung_image, lung_mask)

        else:
            sel_patient_nodules_df = patient_nodules_df[patient_nodules_df['nslice']==nslice]
            regions_pred = extract_rois_from_df(sel_patient_nodules_df)

        # Generate labels
        if np.sum(nodules_mask)!=0:
            regions_real = get_regions(nodules_mask, threshold=np.mean(nodules_mask))
            labels, stats = get_labels_from_regions(regions_real, regions_pred)
        else:
            stats = {'fp':len(regions_pred), 'tp': 0, 'fn':0}
            labels = [0]*len(regions_pred)

        # Extract cropped images
        if thickness>0:  # add extra images as channels for thick resnet
            lung_image = patient_data[0,(nslice - thickness):(nslice + thickness + 1),:,:]
            if lung_image.shape[0] != 2*thickness + 1:  # skip the extremes
                continue
        cropped_images = extract_crops_from_regions(lung_image, regions_pred)


        total_stats = add_stats(stats, total_stats)
        if debug: logging.info("++ Slice %d, stats: %s" % (nslice, str(stats)))

        rois.extend([(nslice, r) for r in regions_pred])  # extend regions with the slice index

        def get_coordinates(roi, nslices):
            x,y = map(lambda x: x / 512 - 0.5, roi[1].centroid)
            z = float(nslice) / (nslices-1) - 0.5
            radi = np.sqrt(x**2+y**2+z**2)
            radi_cilindro = np.sqrt(x**2 + y**2) # esto mas la z, da algo asi como coordenadas cilindricas
            #angle_vert = np.sign(z) * np.arccos(radi_cilindro/radi)
            pos_features = np.array([x,y,z,radi,radi_cilindro])
            return pos_features
        
        X.extend(zip(cropped_images, map(lambda roi: get_coordinates(roi, len(nslices)), rois)))
        Y.extend(labels)  # nodules_mask
    return (X, Y, rois, total_stats) if output_rois else (X, atlas)



