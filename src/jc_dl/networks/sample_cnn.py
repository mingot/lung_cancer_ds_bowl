import numpy as np
from keras.layers.convolutional import (Convolution3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

from utils import net_arch_utils

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


class Sample2DCNNNetworkArchitecture(object):

    def load_model(self, inp_shape):
        model = Sequential()
        # number of convolutional filters to use
        nb_filters = 2
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)
        
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=inp_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        return model

    def __init__(self, inp_shape, use_pretrained = False):
        # This network has no pretrained model... we simply ignore the flag
        self.net = self.load_model(inp_shape)

    def load_weights(self, weights_filename):
        net_arch_utils.load_weights(self.net, weights_filename)

    def save_weights(self, weights_filename):
        net_arch_utils.save_weights(self.net, weights_filename)

    def get_model(self):
        return self.net

    def get_output_shape(self):
        return self.net.layers[-1].output_shape

    def get_input_shape(self):
        return self.net.layers[0].input_shape

