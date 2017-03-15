import itk
import os
import SimpleITK as sitk
import numpy as np
import utils.plotting

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

def getVesselMask(data):
    """

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
