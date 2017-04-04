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


files = ['/mnt/hd2/preprocessed5/luna_102681962408431413578140925249.npz']

batch_size=5
thickness=3
x=thickresunet.uniform_chunks(file_list=files,
    batch_size=batch_size,infinite=True,thickness=thickness,max_slices=6)
                

model = architecture.residual_unet_model((3,512,512))        
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1e-3), 
    loss=thickresunet.weighted_loss, 
    metrics=[thickresunet.weighted_loss,thickresunet.zero_loss])

model.fit_generator(generator=x,
                samples_per_epoch=10,
                nb_epoch=1,
                verbose=1,
                #callbacks=[tb, modchk],
                max_q_size=2,
                nb_worker=1)
