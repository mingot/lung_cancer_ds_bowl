from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D

from utils import net_arch_utils

# U-NET Tutorial Architecture
# REF: https://github.com/booz-allen-hamilton/DSB3Tutorial/blob/master/tutorial_code/LUNA_train_unet.py

class UNETArchitecture(object):

    def _get_UNET(self, inp_shape):
        inputs = Input(inp_shape)

        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        avpooling = AveragePooling2D(pool_size=(512, 512))(conv10)  # final average pooling

        return Model(input=inputs, output=avpooling)  # conv10

    def load_model(self, inp_shape, use_pretrained):
        net = self._get_UNET(inp_shape)
        if use_pretrained:
            self.load_weights('unet_official.hdf5')
        return net

    def __init__(self, inp_shape, use_pretrained):
        self.net = self._get_UNET(inp_shape)
        if use_pretrained:
            self.load_weights('unet_official.hdf5')

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
