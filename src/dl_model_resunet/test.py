import os
import random
import sys
sys.path.append('../')

import os
import random
from thickresunet import ThickRESUNET
import pandas as pd
import thickresunet
import architecture
import numpy as np

files = ['/mnt/hd2/preprocessed5/luna_102681962408431413578140925249.npz']

patient = np.load(files[0])['arr_0']

tac = thickresunet.normalize(patient[0,73:76,:,:])
lungs = patient[1,74,:,:]

print(tac.shape)
print(lungs.shape)

print(np.mean(lungs))
print(np.mean(tac))

def constant_training():
     tmp = np.expand_dims(tac,axis=0)
     X=np.concatenate([tmp,tmp,tmp,tmp,tmp])
     aux = np.expand_dims(lungs,axis=0)
     aux = np.expand_dims(aux,axis=0)
     Y=np.concatenate([aux,aux,aux,aux,aux])
     while True:
         yield X,Y

model = architecture.residual_unet_model((3,512,512))
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1e-3),
     loss=thickresunet.weighted_loss,
     metrics=[thickresunet.weighted_loss,thickresunet.zero_loss])


model.fit_generator(generator=constant_training(),
    samples_per_epoch=100,
    nb_epoch=3,
    verbose=1,
    #callbacks=[tb, modchk],
    max_q_size=2,
    nb_worker=1)

model.save('test.hdf5')


pred=model.predict(np.asarray([thickresunet.normalize(tac)]))
print '--------------------'
print(np.mean(pred))
print(np.max(pred))

raise Exception()

