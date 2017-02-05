# Lung Cancer Data Science Bowl 2017

## References quick start

```
# from the DSB root folder
export LUNG_PATH=`pwd`
cd src/jc_dl
THEANO_FLAGS='device=cpu' python experiments/jose_cordero_sample_experiment/train_network.py
```
Please, notice that I used theano and cpu. This is because the GPU in my computer raised an error if I use any other configuration. Please, feel free to modify it to use tensorflow backend and pu if you prefer so!

## General guidelines
 - Networks should be implemented in 'networks' folder following the sample_iface interface proposed. This is important for reusability. In case your architecture do not have pretrained weights, just ignore that option. The save/load weights functionality is mandatory to be able to train and test the network properly. You can find a couple of examples within that folder:
 -- c3d: C3D facebook's convolution neural network. This example shows how to deal with a pretrained NN
 -- sample_cnn: simple 2D CNN to show how to implement an arbitrary architecture with no pretrained weights.
 - Files (weights and pretrained models) must be saved in the `models` folder of the repo. If possible, io functions must be in a common utils file

## TODO list
 - We must find a way to load datasets. This step must also load the required labels and a way to match train/test cases in order to export the final submission file. This step will be easier when we have an available dataset. It would be super nice to create a simple interface to make the process as painless as possible.
 - Create more networks to be used with different architectures and outputs (example: googlenet, unet...)

## File structure (DL subproject folders)


```
├── README.md          <- The top-level README for DL-developers using this project.
├── networks           <- This folder will contain different network architectures
│   ├── unit_test      <- Folder containing tests for each one of the networks (not required... but ,
│                         recommended). A test file must be an executable which raises an exception if
│                         something doesn't go as expected.
│
├── utils              <- Folder containing common functions which can be reused from different
│                         network architectures
│
├── datasets           <- [TODO] Classes used to manage the different datasets (3D, slices, patches...)
│
├── experiments        <- This folder contains the different tests and experiments. Each one must use as
                          much code as possible from parent folders and nothing from other experiments.
                          Each experient must be self-contained (visualiazation, training and testing)
```

## Troubleshoot

### Sometimes changing the backend of keras from tensorflow to theano generates some problems due to the way both backends read the data.

A solution is to edit the file `~/.keras/keras.json` and make it look as follows:
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```
