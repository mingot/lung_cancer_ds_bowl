import cv2
import keras.preprocessing.image
import scipy.ndimage.interpolation
import numpy as np
import itertools
import os
import collections


class Transformations(object):
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
        return dx, dy
    
    @staticmethod
    def elastic_transform(input_pixels, affine_transform, displacement_field, border_mode = cv2.BORDER_REFLECT_101):
        shape = input_pixels.shape
        output = cv2.warpAffine(input_pixels,
                                            affine_transform,
                                            shape,
                                            borderMode = border_mode)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        dx, dy = displacement_field
        
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        output = (scipy.ndimage.interpolation
                        .map_coordinates(output, indices, order = 1, mode = 'reflect')
                        .reshape(shape))
        
        return output


class VpAugmentedGenerator(object):
    def __init__(self, augmentation_factor, shape, transformations_params, seed):
        self.augmentation_factor = augmentation_factor
        self.shape = shape
        self.transformations_params = transformations_params
        self.seed = seed
    
    @staticmethod
    def transform_patient(p, transformation):
        types, slices, _, _ = p.shape
        trans_p = np.copy(p)
        for type, slice in itertools.product(xrange(types), xrange(slices)):
            trans_p[type, slice] = transformation(trans_p[type, slice])
        
        return trans_p
    
    def generate_transformation(self):
        # Decide if the transformation is the identity
        if np.random.random_sample(1)[0] < 1.0 / self.augmentation_factor:
            return lambda input_pixels: input_pixels
        
        # Generate a random transformation
        tp = np.random.choice(self.transformations_params, size = 1)[0]
        at = Transformations.generate_affine_transform(self.shape, tp['affine']['alpha'])
        dp = Transformations.generate_displacement_field(self.shape, tp['displacement_field']['alpha'], tp['displacement_field']['sigma'])
        return lambda input_pixels: Transformations.elastic_transform(input_pixels, at, dp)
    
    def flow(self, X, batch_size = 16, allow_smaller_batch_size = False):
        batch = collections.deque()
        for augmentation_step in xrange(self.augmentation_factor):
            for k in np.random.permutation(len(X)):
                transformation = self.generate_transformation()
                transformed_patient = self.transform_patient(X[k], transformation)
                _, slices, _, _ = transformed_patient.shape
                start = len(batch)
                while start < slices:
                    delta = batch_size - len(batch)
                    for index in xrange(start, min(start + delta, slices)):
                        batch.append(transformed_patient[:, index, :, :])
                    
                    if len(batch) == batch_size:
                        yield list(batch)
                        batch = collections.deque()
                    
                    start += delta
        
        if len(batch) > 0:
            if (len(batch) < batch_size) and allow_smaller_batch_size:
                yield list(batch)

