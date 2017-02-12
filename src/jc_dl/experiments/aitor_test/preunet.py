import os
import numpy as np
from keras import backend as K
from networks.unet import UNETArchitecture

K.set_image_dim_ordering('th')

input_path = '/mnt/hd2/preprocessed2'

def load_patient(filepath):
    X = []
    b = np.load(os.path.join(input_path,filepath))['arr_0']
    for i in range(b.shape[1]):
       X.append(b[0,i,:512,:512])
    return X

arch = UNETArchitecture((1,512,512),True)

mylist = os.listdir(input_path)
file_list = [g for g in mylist if g.startswith('dsb_')]

net = arch.get_model()

for f in file_list:
    X, Y = load_patient(f), []

    for s in X:
        X = np.expand_dims(np.expand_dims(s,axis=0),axis=0)
        Y.append(net.predict(X)[0,0])
