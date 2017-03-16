import numpy as np
import os
from vpaugmentedgenerator import VpAugmentedGenerator

directory = '/mnt/hd2/preprocessed5/'

shape = (512, 512)
transformations_params = [{'affine': {'alpha': 20}, 'displacement_field': {'alpha': 512, 'sigma': 20}},
                            {'affine': {'alpha': 50}, 'displacement_field': {'alpha': 512, 'sigma': 50}},
                            {'affine': {'alpha': 20}, 'displacement_field': {'alpha': 0, 'sigma': 20}},
                            {'affine': {'alpha': 50}, 'displacement_field': {'alpha': 0, 'sigma': 50}}]


g = VpAugmentedGenerator(4, shape, transformations_params, 19950510)
X = [np.load(os.path.join(directory, filename))['arr_0'] for filename in os.listdir(directory)[:3]]
f = g.flow(X, batch_size = 32)

batch = f.next()

