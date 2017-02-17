import os
import numpy as np
from keras import backend as K
from networks.unet import UNETArchitecture
# from utils.buffering import buffered_gen_mp, buffered_gen_threaded


# A dataset class implements a generator function "generate_data" which returns:
#   X      : data for a given patient or slice depending on the implementation
#   Y      : general label for sample X (i.e. cancer or not cancer, male female... )
#   Y_mask : ROI mask for each X
#   is_valid: a boolean flag indicating if the chunk of data is or is not valid. This should be deleted at some moment... it is just a hack to avoid the class crashing in some corner cases
class Generic_SlicesDataset(object):

    def __init__(self, prefixes_to_load, input_paths):
        K.set_image_dim_ordering('th')
        files_to_load = self._get_files_to_be_loaded_(prefixes_to_load, input_paths)
        self.files_to_load_train = files_to_load[:-1]
        self.files_to_load_valid = files_to_load[-1:]

    def _data_generator_(self, data_set, normalize):
        files_to_load = self.files_to_load_valid if data_set == 'valid' else self.files_to_load_train
        for filepath in files_to_load:
            for X, Y_mask, Y in self._load_patient_from_path_(filepath):
                yield normalize(X), Y_mask, Y

    # Function we can use to ignore a slice.
    def filter_sample(self, X, Y_mask):
        return False

    def get_data(self, data_set, max_batch_size, normalize):
        # verify that the set of data exists
        if data_set not in ('train', 'valid'):
            raise Exception("You must ask for 'train' or 'valid' set")

        # intitialize buffers
        X, Y_mask, Y = [], [], []
        current_elements = 0

        for X_, Y_m_, Y_ in self._data_generator_(data_set, normalize):
            # if the buffer is full, return these values and restart the buffers
            if current_elements == max_batch_size:
                is_valid = len(X) > 0
                if is_valid:
                    yield True, (np.array(X).reshape((len(X), 1,X[0].shape[0], X[0].shape[1])), np.array(Y_mask).reshape((len(Y_mask),1,Y_mask[0].shape[0], Y_mask[0].shape[1])), Y)
                else:
                    yield False, (None, None, None)

                X, Y_mask, Y = [], [], []
                current_elements = 0
            # fill the buffer
            else:
                if not self.filter_sample(X_, Y_m_):
                    X.append(X_)
                    Y_mask.append(Y_m_)
                    Y.append(Y_)
                    current_elements += 1
        is_valid = len(X) > 0
        if is_valid:
            yield True, (np.array(X).reshape((len(X), 1,X[0].shape[0], X[0].shape[1])), np.array(Y_mask).reshape((len(Y_mask),1,Y_mask[0].shape[0], Y_mask[0].shape[1])), Y)
        else:
            yield False, (None, None, None)

    def _get_files_to_be_loaded_(self, prefixes_to_load, input_paths):
        file_list = []
        for inpath_ in input_paths:
            mylist = os.listdir(inpath_)
            for prefix_ in prefixes_to_load:
                file_list += [inpath_+'/'+g for g in mylist if g.startswith(prefix_)]
        return file_list

    def _load_patient_from_path_(self, filepath):
        b = np.load(filepath)['arr_0']
        for i in range(b.shape[1]):
            yield b[0,i,:512,:512], None, None

## This dataset returns Luna dataset (data and masks)
class LunaMasked_SlicesDataset(Generic_SlicesDataset):

    # We overwrite the function to load a specific patient data to load the mask for this specific format dataset
    def _load_patient_from_path_(self, filepath):
        b = np.load(filepath)['arr_0']

        for j in range(b.shape[1]):
            if b.shape[0]!=3:
                continue
            else:
                yield b[0,j,:,:]*b[1,j,:,:], b[2,j,:,:], None  # apply lung mask
                #yield b[0,j,:,:], b[1,j,:,:], None  # apply lung mask



## This dataset returns Luna datasets and masks
class LunaNonEmptyMasked_SlicesDataset(LunaMasked_SlicesDataset):

    # We ignore those samples where there is no mask at all
    def filter_sample(self, X, Y_mask):
        return Y_mask.sum() == 0


if __name__ == "__main__":
    max_batch_size = 10
    #prefixes_to_load = ['dsb_']
    prefixes_to_load = ['luna_']
    input_paths = ['../../data/sample_data']#/mnt/hd2/preprocessed2']

    dataset = LunaNonEmptyMasked_SlicesDataset(prefixes_to_load, input_paths)

    for is_valid, (X, Y_mask, Y) in dataset.get_data('train', max_data_chunk):
        if is_valid:
            ## USE X, Y_mask AND Y WILL BE TRAINING SAMPLES
            pass

    for is_valid, (X, Y_mask, Y) in dataset.get_data('valid', max_data_chunk):
        if is_valid:
            ## USE X, Y_mask AND Y WILL BE VALIDATION SAMPLES
            pass

