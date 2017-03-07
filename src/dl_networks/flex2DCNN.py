import numpy as np
from keras.models import Sequential
from utils import net_arch_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Flatten

# Flexible 2D-CNN architecture
# INPUT -> [[CONV -> RELU]*N -> POOL]*M -> [FC -> RELU]*K -> FC
# Parameters [M,N,K,S,F] (M, N > 0)
# S: Conv Filters Size
# F: # of Filters
# REF: http://cs231n.github.io/convolutional-networks/#architectures

class Flex2DCNN(object):

  def add_convblock(self,model):
    model.add(ZeroPadding2D(padding=(self.P,self.P)))
    model.add(Convolution2D(self.F,self.S,self.S,border_mode='same',activation='relu'))

    return model

  def load_model(self,inp_shape):
    model = Sequential()

    # FIRST CONV BLOCKS
    model.add(ZeroPadding2D(padding=(self.P,self.P),input_shape=inp_shape))
    model.add(Convolution2D(self.F,self.S,self.S,border_mode='same',activation='relu'))
    for i in range(self.N-1):
      model = self.add_convblock(model)
    model.add(MaxPooling2D((2,2),strides=None,border_mode='valid'))

    # CONV BLCOKS
    for i in range(self.M-1):
      for j in range(self.N):
         model = self.add_convblock(model)
      model.add(MaxPooling2D((2,2),strides=None,border_mode='valid'))

    model.add(Flatten())

    # FULLY CONNECTED LAYERS
    for i in range(self.K):
      model.add(Dense(model.layers[-1].output_shape[1],activation='relu'))

    # FINAL LAYER
    model.add(Dense(1,activation='softmax'))

    return model

  def __init__(self,inp_shape,param,use_pretrained=False):
    self.M = param[0]
    self.N = param[1]
    self.K = param[2]
    self.S = param[3]
    self.P = (self.S-1)/2
    self.F = param[4]

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
