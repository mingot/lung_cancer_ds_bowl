import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from utils import plotting
from dl_networks.thickunet import ThickUNET
from dl_utils.heatmap import extract_regions_from_heatmap



wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5/'  # wp + 'data/preprocessed5_sample/'
MODEL_FILE = wp + 'models/thickunet16.hdf5'
OUTPUT_CSV = wp + 'output/noduls_thickunet_v01.csv'

file_list = os.listdir(INPUT_PATH)
file_list = [g for g in file_list if g.startswith('luna_')]
model = ThickUNET(dropout=True, initialdepth=16, input_shape=(5,512,512), saved_file=MODEL_FILE)


# ## Single test
# sel_file = INPUT_PATH + file_list[7]
# patient = np.load(sel_file)['arr_0']
# patient.shape
#
# nstart = time()
# pred = model.predict(sel_file,batch_size=10) # 174s
# print time()-nstart
#
# for i in range(patient.shape[1]):
#     if np.sum(patient[2,i])!=0:
#         print i
#
# nslice = 41
# plotting.plot_mask(patient[0,nslice], patient[2,nslice], 0.8*np.max(patient[2,nslice]))
# plotting.plot_mask(patient[0,nslice], pred[nslice,0], 0.8*np.max(pred[nslice,0]))
# plt.imshow(pred[nslice,0])
# plt.show()




with open(OUTPUT_CSV, 'w') as file:

    # write the header
    file.write('filename,nslice,x,y,diameter,max_intensity,min_intensity,mean_intensity\n')

    for idx, filename in enumerate(file_list):
        tstart = time()

        sel_file = os.path.join(INPUT_PATH, filename)
        patient = np.load(sel_file)['arr_0']
        pred = model.predict(sel_file,batch_size=10) # 174s

        print 'Patient: %s (%d/%d)' % (filename, idx, len(file_list))
        for nslice in range(patient.shape[1]):
            heatmap = pred[nslice,0]
            regions_pred = extract_regions_from_heatmap(heatmap, threshold=0.8*np.max(heatmap))
            for r in regions_pred:
                file.write('%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,
                                                           r.max_intensity, r.min_intensity, r.mean_intensity))
        print time()-tstart

