import cv2
import numpy as np
import scipy.ndimage.filters
import scipy.ndimage.interpolation
from dicomtransformer import DicomTransformer


class DicomTransformerElastic(DicomTransformer):
    def __init__(self, input_path, output_path, affine_transform = None, alpha_affine = None, displacement_field = None, alpha = None, sigma = None, unavailable_value = -2000):
        super(DicomTransformerElastic, self).__init__(input_path, output_path, unavailable_value)
        
        self.alpha = alpha
        self.sigma = sigma
        
        # Affine transform
        if affine_transform is None:
            if alpha_affine is None:
                raise ValueError('alpha_affine must not be None if affine_transform is None')
            
            shape_size = self._input_pixels.shape[:2]
            affine_transform = self.generate_affine_transform(shape_size, alpha_affine)
        else:
            if not(alpha_affine is None):
                print 'WARNING: alpha_affine is ignored if affine_transform is not None'
        
        self.affine_transform = affine_transform
        
        # Displacement field
        if displacement_field is None:
            if alpha is None:
                raise ValueError('alpha must not be None if displacement_field is None')
            
            if sigma is None:
                raise ValueError('sigma must not be None if displacement_field is None')
            
            shape = self._input_pixels.shape
            displacement_field = self.generate_displacement_field(shape, alpha, sigma)
        else:
            if not(alpha is None):
                print 'WARNING: alpha is ignored if displacement_field is not None'
            
            if not(sigma is None):
                print 'WARNING: sigma is ignored if displacement_field is not None'
        
        self.displacement_field = displacement_field
    
    @staticmethod
    def generate_affine_transform(shape_size, alpha_affine, random_state = None):
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                                    [center_square[0] + square_size, center_square[1] - square_size],
                                    center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size = pts1.shape).astype(np.float32)
        return cv2.getAffineTransform(pts1, pts2)
    
    @staticmethod
    def generate_displacement_field(shape, alpha, sigma, random_state = None):
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        dx = scipy.ndimage.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = scipy.ndimage.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)
        return dx, dy, dz
    
    def transform(self):
        shape = self._input_pixels.shape
        shape_size = shape[:2]
        
        output = cv2.warpAffine(self._input_pixels,
                                            self.affine_transform,
                                            shape_size[::-1],
                                            borderMode = cv2.BORDER_REFLECT_101)
        
        # x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        dx, dy, dz = self.displacement_field
        
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        output = (scipy.ndimage.interpolation
                        .map_coordinates(output, indices, order = 1, mode = 'reflect')
                        .reshape(shape))
        
        return output