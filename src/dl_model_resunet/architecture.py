#----------------------------------------------------------------
from __future__ import absolute_import, division
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    UpSampling2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
K.set_image_dim_ordering('th')


l2regularization = l2(1.e-5) # it was 1.e-4

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    #return Activation("relu")(norm)
    return Activation("relu")(input) # disabling batch normalization


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2regularization)

    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2regularization)

    def f(input):
        activation = _bn_relu(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    upsampling = int(round(residual_shape[ROW_AXIS] / input_shape[ROW_AXIS]))
    
    shortcut = input
    if upsampling>1:
        shortcut = UpSampling2D(size=(upsampling,upsampling))(shortcut)
    
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual_shape[CHANNEL_AXIS],
                                 nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(shortcut)

    return merge([shortcut, residual], mode="sum")


def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filter=nb_filter, init_subsample=init_subsample,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(nb_filter, init_subsample=(1, 1), init_upsample=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=3, nb_col=3,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv1)
        if init_upsample != (1,1):
            residual = UpSampling2D(size=init_upsample)(residual)
        return _shortcut(input, residual)

    return f


def bottleneck(nb_filter, init_subsample=(1, 1),init_upsample=(1,1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=1, nb_col=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=1, nb_col=1,
                                     subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv_1_1)
        if init_upsample != (1,1):
            conv_3_3 = UpSampling2D(size=init_upsample)(conv_3_3)
        residual = _bn_relu_conv(nb_filter=nb_filter * 4, nb_row=1, nb_col=1)(conv_3_3)
        return _shortcut(input, residual)

    return f

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3
        
_handle_dim_ordering()        


def residual_unet_model(inp_shape):
    inputs = Input(inp_shape)


    down1=Convolution2D(32,nb_row=3, nb_col=3,
                        subsample=(1,1),
                        init="he_normal", border_mode="same",
                        W_regularizer=l2(0.0001))(inputs)
    print(down1.get_shape())

    down2=basic_block(32,init_subsample=(2,2))(down1)

    down3= bottleneck(32,init_subsample=(2,2))(down2)
    down3= bottleneck(32,init_subsample=(1,1))(down3)
    down3= bottleneck(32,init_subsample=(1,1))(down3)
    print(down3.get_shape())

    down4= bottleneck(64,init_subsample=(2,2))(down3)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    down4= bottleneck(64,init_subsample=(1,1))(down4)
    print(down4.get_shape())

    down5= bottleneck(128,init_subsample=(2,2))(down4)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    down5= bottleneck(128,init_subsample=(1,1))(down5)
    print(down5.get_shape())

    across=bottleneck(256,init_subsample=(1,1))(down5)
    across=bottleneck(256,init_subsample=(1,1))(across)
    across=bottleneck(256,init_subsample=(1,1))(across)
    print(across.get_shape())
    
    up1 = bottleneck(128,init_upsample=(1,1))(across)
    up1 = bottleneck(128,init_upsample=(1,1))(up1)
    up1 = bottleneck(128,init_upsample=(1,1))(up1)
    up1 = bottleneck(128,init_upsample=(1,1))(up1)
    up1 = bottleneck(128,init_upsample=(1,1))(up1)
    up1 = bottleneck(128,init_upsample=(1,1))(up1)
    up1 = bottleneck(128,init_upsample=(1,1))(up1)
    up1 = bottleneck(128,init_upsample=(2,2))(up1)
    print(up1.get_shape())

    up1=_shortcut(down4, up1)
    print(up1.get_shape())

    up2 = bottleneck(64,init_upsample=(1,1))(up1)
    up2 = bottleneck(64,init_upsample=(1,1))(up2)
    up2 = bottleneck(64,init_upsample=(2,2))(up2)
    print(up2.get_shape())

    up2=_shortcut(down3, up2)
    print(up2.get_shape())

    up3 = bottleneck(32,init_upsample=(2,2))(up2)
    print(up3.get_shape())

    up3=_shortcut(down2, up3)
    print(up3.get_shape())

    up4 = basic_block(32,init_upsample=(2,2))(up3)
    print(up4.get_shape())

    up4=_shortcut(down1, up4)
    print(up4.get_shape())

    up5 = Convolution2D(32,nb_row=3, nb_col=3,
                        subsample=(1,1),
                        init="he_normal", border_mode="same",
                        W_regularizer=l2(0.0001))(up4)

    print(up5.get_shape())

    output = Convolution2D(1, 1, 1, activation='sigmoid')(up5)

    print(output.get_shape())
    
    model=Model(input=inputs, output=output)
    return(model)

    
