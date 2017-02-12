import numpy as np

from networks.flex2DCNN import Flex2DCNN

SIZE = 10

# DUMMY DATA
data = np.random.random((SIZE,100,512,512))
labels = np.random.randint(2, size=(SIZE))

arch = Flex2DCNN((100,512,512),[2,1,0,3,1])
model = arch.get_model()

model.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data, labels, nb_epoch=2, batch_size=1)