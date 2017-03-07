import numpy as np
from keras.layers.convolutional import (Convolution3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

from utils import net_arch_utils

# You can look into the "c3d_network" file to see an example of how a network can be actually implemented.

class SampleIfaceNetworkArchitecture(object):
    def __init__(self, inp_shape, use_pretrained = False):

    def load_weights(self, weights_filename):

    def save_weights(self, weights_filename):

    def get_model(self):

    def get_output_shape(self):

    def get_input_shape(self):

