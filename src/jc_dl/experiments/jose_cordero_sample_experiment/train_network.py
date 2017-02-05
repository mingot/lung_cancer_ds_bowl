# for custom metrics
import numpy as np
import keras.backend as K


## Dummy data loader. This should be encapsulates within nice classes
def load_dummy_data(net_name):
    sample_size = 10
    if net_name == 'c3d':
        data = np.random.random((sample_size, 3,16,112,112))
        labels = np.random.randint(2, size=(sample_size, 1))
    elif net_name == 'sample_2d_cnn':
        data = np.random.random((sample_size, 1, 50,50))
        labels = np.random.randint(2, size=(sample_size, 1))
    return data,labels

## Helper to load and test each network. For a real experiment, we will only load one... so this will be much cleaner
def load_network(net_name):
    if net_name == 'c3d':
        from networks.c3d_network import C3DNetworkArchitecture
        netarch = C3DNetworkArchitecture((3,16,112,112), use_pretrained = True)
    elif net_name == 'sample_2d_cnn':
        from networks.sample_cnn import Sample2DCNNNetworkArchitecture
        netarch = Sample2DCNNNetworkArchitecture((1,50,50), use_pretrained = True)
    if not netarch:
        raise Exception("The network `%s` could not be loeaded" % net_name)

    return netarch.get_model()

## Once we have the model and the data, we simply fit it!
def train_model(model, X, Y):
    model.compile(  optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                  )
    model.fit(X, Y, nb_epoch=10, batch_size=5)

if __name__ == '__main__':
    net_to_test = 'sample_2d_cnn'

    model = load_network(net_to_test)
    X,Y = load_dummy_data(net_to_test)
    train_model(model, X, Y)
    #test_model(model)