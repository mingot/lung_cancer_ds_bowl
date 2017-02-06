import numpy as np
import os

# see src/00_preprocess.py and in particular
# np.savez_compressed(os.path.join(OUTPUT_FOLDER, "%s_%s.npz") % (PIPELINE, pat_id), output)

DATASET_DSB = 'dsb'
# DATASET_LUNA = 'luna'
# DATASET_LIDC = 'lidc'


# see README.md Preprocessing
# returns a numpy array of 5 dimensions: [patient_number,type,slice,height,width]
# where patient_number is between 0 and number_of_patients
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
		b = np.load(filename)['arr_0']
		ret.append(b)
	return np.asarray(ret)

if __name__ == '__main__':
	l = load(DATASET_DSB,1,5)
