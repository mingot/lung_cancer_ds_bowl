import numpy as np
from keras.layers.convolutional import (Convolution3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

from utils import net_arch_utils

class C3DNetworkArchitecture(object):

    def _get_c3d_model(self, inp_shape):
        if not (inp_shape == np.array((3,16,112,112))).all():
            raise Exception("Input shape is not comptible with the pretrained C3D model. Are you sure you want to use this network?")

        c3d_model = Sequential()
        c3d_model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv1',
                                subsample=(1, 1, 1),
                                input_shape=(3, 16, 112, 112)))
        c3d_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        c3d_model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2',
                                subsample=(1, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        c3d_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a',
                                subsample=(1, 1, 1)))
        c3d_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b',
                                subsample=(1, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a',
                                subsample=(1, 1, 1)))
        c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b',
                                subsample=(1, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a',
                                subsample=(1, 1, 1)))
        c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b',
                                subsample=(1, 1, 1)))
        c3d_model.add(ZeroPadding3D(padding=(0, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        c3d_model.add(Flatten())
        c3d_model.add(Dense(4096, activation='relu', name='fc6'))
        c3d_model.add(Dropout(.5))
        c3d_model.add(Dense(4096, activation='relu', name='fc7'))
        c3d_model.add(Dropout(.5))
        c3d_model.add(Dense(487, activation='softmax', name='fc8'))
        return c3d_model

    def load_model(self, inp_shape, use_pretrained):
        net = self._get_c3d_model(inp_shape)
        if use_pretrained:
            filepath = net_arch_utils.download_file('c3d_weights.h5', self._C3D_WEIGHTS_URL)
            net.load_weights(filepath)
        net.layers.pop()
        net.add(Dense(1, activation='tanh', name='new_output_layer'))
        return net

    def __init__(self, inp_shape, use_pretrained = False):
        self._C3D_WEIGHTS_URL = 'https://www.dropbox.com/s/ypiwalgtlrtnw8b/c3d-sports1M_weights.h5?dl=1'
        self.net = self.load_model(inp_shape, use_pretrained)

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

