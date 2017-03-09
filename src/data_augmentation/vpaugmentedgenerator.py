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
    def __init__(self, shape, transformations_params, seed):
        self.shape = shape
        self.transformations_params = transformations_params
        self.seed = seed
        self._total_batches_seen = 0
    
    @staticmethod
    def patient_augmenter(p, transformations):
        types, slices, _, _ = p.shape
        for transformation in transformations:
            trans_p = np.copy(p)
            for type, slice in itertools.product(xrange(types), xrange(slices)):
                trans_p[type, slice] = transformation(trans_p[type, slice])
            
            yield trans_p
    
    def generate_transformations(self):
        affine_transforms = [Transformations.generate_affine_transform(self.shape, tp['affine']['alpha']) for tp in self.transformations_params]
        displacement_fields = [Transformations.generate_displacement_field(self.shape, tp['displacement_field']['alpha'], tp['displacement_field']['sigma']) for tip in self.transformations_params]
        transformations = [None for k in xrange(len(zip(affine_transforms, displacement_fields)))]
        for k, (at, dp) in enumerate(zip(affine_transforms, displacement_fields)):
            transformations[k] = lambda input_pixels: Transformations.elastic_transform(input_pixels, at, dp)
        
        return transformations
    
    def flow(self, X, batch_size = 8):
        for _, batch in itertools.groupby(enumerate(X), key = lambda elem: elem[0] // batch_size):
            np.random.seed(self.seed + self._total_batches_seen)
            augmented_patients = collections.deque()
            for _, p in batch:
                transformations = self.generate_transformations()
                
                for ap in self.patient_augmenter(p, transformations):
                    augmented_patients.append(ap)
                
            self._total_batches_seen += 1
            yield list(augmented_patients)
    
    def flow_from_directory(self, directory, batch_size = 8):
        for _, batch in itertools.groupby(enumerate(os.listdir(directory)), key = lambda elem: elem[0] // batch_size):
            np.random.seed(self.seed + self._total_batches_seen)
            augmented_patients = collections.deque()
            for _, filename in batch:
                transformations = self.generate_transformations()
                
                for ap in self.patient_augmenter(np.load(os.path.join(directory, filename))['arr_0'], transformations):
                    augmented_patients.append(ap)
            
            self._total_batches_seen += 1
            yield list(augmented_patients)
