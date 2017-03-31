directory = '/mnt/hd2/preprocessed5/'

shape = (512, 512)
transformations_params = [{'affine': {'alpha': 20}, 'displacement_field': {'alpha': 512, 'sigma': 20}},
                 	       {'affine': {'alpha': 50}, 'displacement_field': {'alpha': 512, 'sigma': 50}},
                 	       {'affine': {'alpha': 20}, 'displacement_field': {'alpha': 0, 'sigma': 20}},
                                {'affine': {'alpha': 50}, 'displacement_field': {'alpha': 0, 'sigma': 50}}]


g = VpAugmentedGenerator(shape, transformations_params, 19950510)
f = g.flow_from_directory(directory, batch_size = 8)

X = f.next()

