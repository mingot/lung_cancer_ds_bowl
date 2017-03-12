# VP (Vila del Pingui) Data Augmented Generator

# VpAugmentedGenerator class
VpAugmentedGenerator class comes with two methods, `flow` and `flow_from_directory`, to construct generators to provide an augmented dataset based on elastic random deformations.

## Methods
### \_\_init\_\_(shape, transformations\_params, seed)
Args:

- `augmentation_factor`: number of times the data will be augmented

- `shape`: The shape of the images. Ex: `(512, 512)`

- `transformations_params`: a list of transformations params. The transformations applied to each patient will be randomly drawn from this list. Each element is a dictionary of params. Ex: `{'affine': {'alpha': 20}, 'displacement_field': {'alpha': 512, 'sigma': 20}}`

- `seed`

### generate\_transformation()
For internal use. Generates a random transformation to be applied to a patient. It generates the identity transformation with probability `1 / augmentation_factor`, and a random elastic deformation otherwise.

### flow(X, batch\_size)
Takes a set of patients and returns a generator to provide augmented data.

Args:

- `X`: a list of patients loaded in memory. This list will be randomly iterated a number of times equal to the `augmentation_factor`.

- `batch_size`: number of slices to be provided in each iteration. Defaults to `16`.

- `allow_smaller_batch_size`: allow the generator to return a batch smaller than `batch_size`. This may happen in the last iteration.

## Example
```python
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
```
