import itk
import os
import SimpleITK as sitk
import numpy as np
from ..utils.reading import get_number_of_nodules

def count_number_of_lost_nodules(nodules_slice_mask, vessel_mask):
    """
    Counts the number of nodules lost during the process of calculating the vessel mask
    :param nodules_slice_mask: nodules pre mask
    :param vessel_mask: vessel mask
    :return: and Integer, the number of nodules remaining after calculating the Vessel Mask
    """
    subs = nodules_slice_mask - vessel_mask
    subs[subs<0] = 0
    return get_number_of_nodules(subs)

def count_nodules(nodules_slices, vessel_mask):
    """
    Counts the number of nodules in all the slices of the patient, before and after the vessel mask
    :param nodules_slices: list of all the nodules slices of the patient
    :param vessel_mask: list of all the vessel masks of the patient
    :return: A tuple with the number of nodules at the beginning and after aplying the mask
    """
    original, processed = 0, 0
    for n, v in zip(nodules_slices, vessel_mask):
        original += get_number_of_nodules(n)
        processed += count_number_of_lost_nodules(n, v)
    return original, processed

def binarize_image(im, threshold):
    """
    Given a heatmap, returns a binary image
    :param im: the image to binarize
    :param threshold: threshold
    :return: a binary image
    """
    im[im < threshold] = 0
    im[im >= threshold] = 1
    return im

def get_vessel_mask(data):
    """
    Given the list of slices of the patient, returns the list of its Vessels Mask
    :param data: a numpy array containig all the patient slides. The result of doing np.load(...)['arr_0'][0]
    :return: an array containing the vessels mask for each slide of the patient
    """
    InputPixelType = itk.F
    OutputPixelType = itk.UC
    Dimension = 3
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]

    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]

    VesselnessFilterType = itk.Hessian3DToVesselnessMeasureImageFilter[itk.F]
    MultiScaleEnhancementFilterType = itk.MultiScaleHessianBasedMeasureImageFilter[
        InputImageType, HessianImageType, InputImageType]

    simage = sitk.GetImageFromArray(data.astype('float32'))
    lung_float = data.astype('float32')
    image = itk.PyBuffer[InputImageType].GetImageFromArray(lung_float)

    # Instantiate the multiscale filter and set the input image
    multiScaleEnhancementFilter = MultiScaleEnhancementFilterType.New()
    multiScaleEnhancementFilter.SetInput(image)
    multiScaleEnhancementFilter.SetSigmaMinimum(0.5)
    multiScaleEnhancementFilter.SetSigmaMaximum(4.0)
    multiScaleEnhancementFilter.SetNumberOfSigmaSteps(5)

    # Get the vesselness filter and set the parameters
    vesselnessFilter = VesselnessFilterType.New()
    vesselnessFilter.SetAlpha1(0.5)
    vesselnessFilter.SetAlpha2(0.5)
    multiScaleEnhancementFilter.SetHessianToMeasureFilter(vesselnessFilter)

    RescaleFilterType = itk.RescaleIntensityImageFilter[InputImageType, OutputImageType]
    rescaleFilter = RescaleFilterType.New()
    rescaleFilter.SetInput(multiScaleEnhancementFilter.GetOutput())

    rescaleFilter.Update()

    bin_image = binarize_image(itk.PyBuffer[OutputImageType].GetArrayFromImage(rescaleFilter.GetOutput()), 25)

    return bin_image


if __name__ == "__main__":
    path = '/mnt/hd2/preprocessed5_sample/'
    for file in os.listdir(path):
        print("Loading the patient...")
        patient = np.load(path + file)['arr_0']
        print("Calculating its vessel mask...")
        vessel_mask = get_vessel_mask(patient[0])
        if patient.shape[0] < 3: #does not have a nodules_slice_mask
            print("The patient does not have a nodules slice mask")
            continue
        print("Counting nodules...")
        original_nodules, post_mask_nodules = count_nodules(patient[2], vessel_mask)
        print("Patient: " + file.split('/')[-1])
        print("Original nodules: \t\t\t\t" + str(original_nodules))
        print("Number of nodules afther aplying the mask: " + str(post_mask_nodules))
