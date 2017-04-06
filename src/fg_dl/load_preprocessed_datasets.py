import numpy as np
import os
import os.path

# see src/preprocess.py and in particular
# np.savez_compressed(os.path.join(OUTPUT_FOLDER, "%s_%s.npz") % (PIPELINE, pat_id), output)

DATASET_DSB = 'dsb'
# DATASET_LUNA = 'luna'
# DATASET_LIDC = 'lidc'


# see README.md Preprocessing
# returns an array of dimensions number_of_patients whose elements are
# numpy array of 4 dimensions: [type,slice,height,width]
def load(dataset, index, number_of_patients):
	def get_path_dataset(mydataset):
		if mydataset == DATASET_DSB:
			return '/mnt/hd2/preprocessed/'
	mypath = get_path_dataset(dataset)
	mylist = os.listdir(mypath)
	if dataset == DATASET_DSB:
		file_list = [g for g in mylist if g.startswith('dsb_')]
	file_list.sort()
	file_list = file_list[index: index + number_of_patients]
	ret = []
	for filename in file_list:
		b = np.load(os.path.join(mypath,filename))['arr_0']
		ret.append(b)
	return ret

if __name__ == '__main__':
	l = load(DATASET_DSB,0,1)
	element = l[0]
	print element.shape
