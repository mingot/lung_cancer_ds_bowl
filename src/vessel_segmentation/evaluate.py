import os
import sys
import itk
import SimpleITK as sitk
sys.path.append('/home/marti/lung_cancer_ds_bowl/src')
sys.path.append('/home/marti/lung_cancer_ds_bowl/src/vessel_segmentation')
import numpy as np
import random
from skimage.morphology import disk, binary_erosion, binary_closing, dilation
from dl_model_patches.common import load_patient, add_stats
from vessel_segmentation import get_vessel_mask, substract_from_existing_mask


def subs_dict(d1, d2):
    d = {}
    for k, v in d1.items():
        d[k] = v - d2[k]
    return d


def did_we_lost_full_nodules(tp_per_slice_a, tp_per_slice_b):
    # CONDICIO CUTRE I MILLORABLE
    # We can assure we did not lose full nodules if in at least one slice with tp > 0
    # we have the same number of nodules before and after applying the vessel mask
    for slice, tp in tp_per_slice_a.items():
        if tp > 0 and tp == tp_per_slice_b[slice]:
            return False
    return True


def evaluate(INPUT_PATH, n_patients, dilate=True, binarize_threshold=25, rand_seed=None):
    """

    :param INPUT_PATH: path to the sample of patients
    :param n_patients: number of patients at random that you want to evaluate
    :param dilate: whether you want to dilate the vessel mask or not
    :param binarize_threshold: threshold to convert the mask into a binary mask. Values under
           the threshold will be 0s and values above will be 1s
    :param rand_seed: if you want to fix a seed to always evaluate the same patients
    :return: nothing, just prints
    """
    full_nodules_lost = 0
    file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)]
    if n_patients > len(file_list):
        raise ValueError("You cannot ask for more patients than " + str(len(file_list)))

    if rand_seed is not None:
        random.seed(rand_seed)
    for filename in [file_list[i] for i in random.sample(xrange(len(file_list)), n_patients)]:
        print "Loading patient: " + filename
        patient_data = np.load(filename)['arr_0']
        print "Getting its vessel mask..."
        vessel_mask = get_vessel_mask(patient_data[0], binarize_threshold=binarize_threshold)
        if dilate:
            vessel_mask = dilation(vessel_mask)
        # stats before the mask
        X, y, rois, stats_1, tp_a = load_patient(patient_data=patient_data, patient_nodules_df=None,
                                         discard_empty_nodules=True, output_rois=True,
                                         debug=False, thickness=1, debug_vessel=True)

        patient_data[1, :, :, :] = substract_from_existing_mask(patient_data[1, :, :, :], vessel_mask)

        # stats after the mask
        X, y, rois, stats_2, tp_b = load_patient(patient_data=patient_data, patient_nodules_df=None,
                                         discard_empty_nodules=True, output_rois=True,
                                         debug=False, thickness=1, debug_vessel=True)
        # print "Stats after applying the vessel mask:\n", stats
        sub_d = subs_dict(stats_1, stats_2)


        # PRINTING THE RESULTS
        print "Results for patient " + filename
        print "----------------------------------------------"
        print "Lost " + str(sub_d['fp']) + " false positives"
        if sub_d['tp'] != 0:
            print "Lost " + str(sub_d['tp']) + " TRUE POSITIVES"
        else:
            print "Lost " + str(sub_d['tp']) + " true positives"
        if did_we_lost_full_nodules(tp_a, tp_b):
            print "CAREFUL!, WE MIGHT HAVE LOSE SOME NODULE HERE"
            full_nodules_lost += 1
        print "----------------------------------------------"
        print ""
        print ""

    print "After checking " + str(n_patients) + " we lost " + str(full_nodules_lost) + " full nodules"

if __name__ == '__main__':
    # wp = os.environ['LUNG_PATH']
    INPUT_PATH = '/mnt/hd2/preprocessed5_sample'
    # INPUT_PATH = wp + 'data/preprocessed5_sample_watershed'
    # INPUT_PATH = wp + 'data/preprocessed5_sample_th2'
    evaluate(INPUT_PATH=INPUT_PATH, n_patients=5)