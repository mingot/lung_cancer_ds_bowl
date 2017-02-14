#!/usr/bin/python2
import os
import sys

import numpy as np
from keras import backend as K

sys.path.append(os.path.dirname(__file__) + "../jc_dl/")
from networks.unet import UNETArchitecture

def apply_network_single(neural_net_model, weights_file_path, dataset_input_folder, dataset_prefix_string, output_folder, theano_image):
	if theano_image:
		K.set_image_dim_ordering('th')
	neural_net_model.load_weights(weights_file_path)

	if dataset_input_folder == output_folder:
		print "dataset input and output folders must be different"
		return None

	def files():
		# debug = False
		v = os.listdir(dataset_input_folder)
		# if debug: print v
		v2 = list(filter(lambda s: s.startswith(dataset_prefix_string), v))
		# if debug: print v2
		v2.sort()
		# if debug: print v2
		return v2

	file_list = files()

	for filename in file_list:
		b = np.load(os.path.join(dataset_input_folder,filename))['arr_0']
		X = []
		for j in range(b.shape[1]):
			X.append(b[0,j,:,:])
		X = np.expand_dims(np.asarray(X),axis=1)
    		
		Y = neural_net_model.predict(X,batch_size=1, verbose=1)
		# print Y
		np.savez_compressed(os.path.join(output_folder, filename), Y)

	print "success!"
	return True

def felix_test():
	net_obj = UNETArchitecture((1,512,512),False)
	model = net_obj.get_model()
	dataset_input_path = "/home/qk/luna/"
	output_path = "/home/qk/output/"
	weights_file_path = "/home/qk/lung_cancer_ds_bowl/models/unet_official.hdf5"
	apply_network_single(model, weights_file_path, dataset_input_path, "luna_", output_path, True)
	print "success!"
	
if __name__ == "__main__":
	felix_test()
