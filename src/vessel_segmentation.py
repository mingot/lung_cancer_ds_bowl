import itk
import SimpleITK as sitk
import numpy as np
import utils.plotting

data = np.load('../luna_100225287222365663678666836860.npz')['arr_0']

InputPixelType = itk.F
OutputPixelType = itk.UC
Dimension = 3
HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]

InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]
HessianImageType = itk.Image[HessianPixelType, Dimension]

VesselnessFilterType = itk.Hessian3DToVesselnessMeasureImageFilter[itk.F]
MultiScaleEnhancementFilterType = itk.MultiScaleHessianBasedMeasureImageFilter[InputImageType,HessianImageType,InputImageType]

simage = sitk.GetImageFromArray(data[0].astype('float32'))
lung_float = data[0].astype('float32')
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

data = np.concatenate((data, [itk.PyBuffer[OutputImageType].GetArrayFromImage(rescaleFilter.GetOutput())]))

utils.plotting.multiplot(data[3])