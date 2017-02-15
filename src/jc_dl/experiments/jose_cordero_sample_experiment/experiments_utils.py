import os
import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from networks.unet import UNETArchitecture
from datasets.basic_dataset import LunaNonEmptyMasked_SlicesDataset

def visualize_case(X, Y, model):
    from pylab import *
    # We get the slice with a mask in the label
    slice_max_ind = Y.sum(axis=3).sum(axis=2).sum(axis=1).argmax()
    # We predict for that specific slice
    aux = model.predict(X[slice_max_ind:slice_max_ind+1,:,:,:])
    # We plot it
    figure()
    subplot(221)
    imshow(X[slice_max_ind,0], cmap = cm.Greys)
    subplot(222)
    imshow(aux[0,0], cmap = cm.Greys)
    subplot(223)
    imshow(Y[slice_max_ind,0], cmap = cm.Greys)
    show()

# MODEL METHODS
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

