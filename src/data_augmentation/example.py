from wholedicomprocessor import WholeDicomProcessor
from dicomtransformerelastic import DicomTransformerElastic
from generaterotation import generate_rotation
from generatesimmetry import generate_simmetry


input_dir = '/home/bgarcia/Documents/vpdataaugmenter/sample_images/'
dicom_name = '00cba091fa4ad62cc3200a657aeb957e'


# Elastic
elastic_output_dir = '/home/bgarcia/Documents/vpdataaugmenter/staging_deformed_images/elastic/'


## Generate a unique affine transform that will be used for every slice
affine_transform = DicomTransformerElastic.generate_affine_transform(shape_size = (512, 512),
                                                                        alpha_affine = 20.48)

## Generate a unique displacement field that will be used for every slice
displacement_field = DicomTransformerElastic.generate_displacement_field(shape = (512, 512),
                                                                            alpha = 512,
                                                                            sigma = 20.48)

## Instantiate a WholeDicomProcessor class to apply the same transform for every slice
elastic_example = WholeDicomProcessor(input_dir = input_dir,
                                                        output_dir = elastic_output_dir,
                                                        dicom_name = dicom_name,
                                                        TransformerClass = DicomTransformerElastic,
                                                        affine_transform = affine_transform,
                                                        displacement_field = displacement_field)

elastic_example.run()





# Rotation
rotation_output_dir = '/home/bgarcia/Documents/vpdataaugmenter/staging_deformed_images/rotation/'


## Generate a unique rotation that will be used for every slice
rotation = generate_rotation(shape_size = (512, 512), theta = 5)


## Generate a null displacement_field
displacement_field = DicomTransformerElastic.generate_displacement_field(shape = (512, 512),
                                                                                                    alpha = 0,
                                                                                                    sigma = 0)


## Instantiate a WholeDicomProcessorclass to apply the same transform for every slice
rotation_example = WholeDicomProcessor(input_dir = input_dir,
                                                        output_dir = rotation_output_dir,
                                                        dicom_name = dicom_name,
                                                        TransformerClass = DicomTransformerElastic,
                                                        affine_transform = rotation,
                                                        displacement_field = displacement_field)


rotation_example.run()


# Simmetry
simmetry_output_dir = '/home/bgarcia/Documents/vpdataaugmenter/staging_deformed_images/simmetry/'


## Generate a unique simmetry that will be used for every slice
simmetry = generate_simmetry(shape_size = (512, 512), theta = 0)


## Generate a null displacement_field
displacement_field = DicomTransformerElastic.generate_displacement_field(shape = (512, 512),
                                                                                                    alpha = 0,
                                                                                                    sigma = 0)


## Instantiate a WholeDicomProcessorclass to apply the same transform for every slice
simmetry_example = WholeDicomProcessor(input_dir = input_dir,
                                                            output_dir = simmetry_output_dir,
                                                            dicom_name = dicom_name,
                                                            TransformerClass = DicomTransformerElastic,
                                                            affine_transform = simmetry,
                                                            displacement_field = displacement_field)


simmetry_example.run()
