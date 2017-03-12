# VP (Vila del Pingui) Data Augmented Generator

## VpAugmentedGenerator class
VpAugmentedGenerator class comes with two methods, `flow` and `flow_from_directory`, to construct generators to provide an augmented dataset based on elastic random deformations.

### Methods
#### \_\_init\_\_(shape, transformations\_params, seed)
Args:

- `shape`: The shape of the images. Ex: `(512, 512)`

- `transformations_params`: a list of length equal to the number of transformations per patient to be applied, where each element is a dictionary of params. Ex: `{'affine': {'alpha': 20}, 'displacement_field': {'alpha': 512, 'sigma': 20}}`

- `seed`

#### generate\_transformations
For internal use. Generates a set of random transformations to be applied to a patient.

#### flow
Takes a set of patients and returns a generator to provide augmented data.

Args:

- `X`: an iterable containing a set of patients from the pre-processed set

- `batch_size`: number of *input* images to be processed in each iteration. The size of the output batches will be `batch_size` multiplied by the number of transformations. Defaults to `8`.

#### flow\_from\_directory
Args:

- `directory`: a directory containing pre-processed images

- `batch_size`: number of *input* images to be processed in each iteration. The size of the output batches will be `batch_size` multiplied by the number of transformations. Defaults to `8`.

### Example
```python
directory = '/mnt/hd2/preprocessed5/'

shape = (512, 512)
transformations_params = [{'affine': {'alpha': 20}, 'displacement_field': {'alpha': 512, 'sigma': 20}},
                 	       {'affine': {'alpha': 50}, 'displacement_field': {'alpha': 512, 'sigma': 50}},
                 	       {'affine': {'alpha': 20}, 'displacement_field': {'alpha': 0, 'sigma': 20}},
                                {'affine': {'alpha': 50}, 'displacement_field': {'alpha': 0, 'sigma': 50}}]


g = VpAugmentedGenerator(shape, transformations_params, 19950510)
f = g.flow_from_directory(directory, batch_size = 8)

X = f.next()
```
