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


from keras.models import load_model

model = load_model('my_model.hdf5',custom_objects={'weighted_loss':thickresunet.weighted_loss, 'zero_loss':thickresunet.zero_loss})

pred=model.predict(np.asarray([thickresunet.normalize(tac)]))
print(np.mean(pred))
print(np.max(pred))


raise Exception()

batch_size=5
thickness=3
#x=thickresunet.uniform_chunks(file_list=files,
#    batch_size=batch_size,infinite=True,thickness=thickness,max_slices=6)
                

model = architecture.residual_unet_model((3,512,512))        
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1e-3), 
    loss=thickresunet.weighted_loss, 
    metrics=[thickresunet.weighted_loss,thickresunet.zero_loss])

model.fit_generator(generator=x,
                samples_per_epoch=100,
                nb_epoch=4,
                verbose=1,
                #callbacks=[tb, modchk],
                max_q_size=2,
                nb_worker=1)

model.save('my_model.hdf5')
