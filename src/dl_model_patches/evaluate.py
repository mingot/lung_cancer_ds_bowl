import os
import logging
import numpy as np
from time import time
from keras.optimizers import Adam
from dl_networks.sample_resnet import ResnetBuilder
from dl_model_patches import  common
from keras import backend as K


K.set_image_dim_ordering('th')

# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'
NODULES_PATH = wp + 'data/luna/annotations.csv'
OUTPUT_MODEL = wp + 'models/jm_patches_hardnegative_v03.hdf5'

# OUTPUT_CSV = wp + 'output/nodules_patches_dl1_v11_solo_luna.csv'
OUTPUT_CSV = wp + 'output/nodules_patches_TEST.csv'

# OUTPUT_MODEL = wp + 'models/jm_patches_hardnegative_v02.hdf5'
# OUTPUT_CSV = wp + 'output/nodules_patches_dl2_v02.csv'



## Params and filepaths
# NOTA: Cargando validation luna i dsb
THICKNESS = 1
file_list = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH)]
file_list += [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH) if fp.startswith('luna_')] # if fp.startswith('dsb_')]


# # ## if the OUTPUT_CSV file already exists, continue it
# previous_filenames = set()
# if os.path.exists(OUTPUT_CSV):
#     write_method = 'a'
#     with open(OUTPUT_CSV) as file:
#         for l in file:
#             previous_filenames.add(l.split(',')[0])


# Load model
model = ResnetBuilder().build_resnet_50((3,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
logging.info('Loading existing model...')
model.load_weights(OUTPUT_MODEL)

# PARALLEL -------------------------------------------------------------------------------------------------------


import multiprocessing

#
# def load_patient_func(filename):
#     patient_data = np.load(filename)['arr_0']
#     X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, thickness=1)
#     logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
#     return X, y, rois, stats
#
#
# def pred(q,xf,rois):
#     preds = model.predict(xf, verbose=1)
#     logging.info("Batch results: %d/%d (th=0.7)" % (len([p for p in preds if p>0.7]),len(preds)))
#     for i in range(len(preds)):
#         nslice, r = roisf[i]
#         file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (ref_filenames[i].split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],yf[i]))
#
#
# with open(OUTPUT_CSV, 'w') as file:
#     file.write('patientid,nslice,x,y,diameter,score,label\n')
#
#     NUM_CONC = 16
#     for j in range(0, len(file_list), NUM_CONC):
#         filenames = file_list[j:j + NUM_CONC]
#         pool =  multiprocessing.Pool(4)
#         x, y, rois, stats = zip(*pool.map(load_patient_func, filenames))
#         logging.info("Batch %d loaded" % j)
#
#         xf, yf, ref_filenames, roisf = [], [], [], []
#         for i in range(len(x)):
#             ref_filenames.extend([filenames[i]]*len(x[i]))
#             xf.extend(x[i])
#             yf.extend(y[i])
#             roisf.extend(rois[i])
#         pool.close()
#         pool.join()
#
#         xf = np.asarray(xf)
#         preds = model.predict(xf, verbose=1)
#         logging.info("Batch results: %d/%d (th=0.7)" % (len([p for p in preds if p>0.7]),len(preds)))
#         for i in range(len(preds)):
#             nslice, r = roisf[i]
#             file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (ref_filenames[i].split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],yf[i]))


# NON PARALLEL ------------------------------------------------------------------------------------------------------

# with open(OUTPUT_CSV, write_method) as file:
#
#     # write the header if the file is new
#     if write_method=='w':
#         file.write('patientid,nslice,x,y,diameter,score,label\n')
#
#     for idx, filename in enumerate(file_list):
#         if filename in previous_filenames:
#             continue
#
#         tstart = time()
#         logging.info("Patient %s (%d/%d)" % (filename, idx, len(file_list)))
#         try:
#             patient_data = np.load(filename)['arr_0']
#             X, y, rois, stats = common.load_patient(patient_data, output_rois=True, thickness=THICKNESS)
#
#             if len(X)==0:
#                 continue
#
#             X = np.asarray(X)
#             if THICKNESS==0:
#                 X = np.expand_dims(X, axis=1)
#             preds = model.predict(X, verbose=1)
#         except:
#             logging.info("Error in patient %s, skipping" % filename)
#             continue
#
#         for i in range(len(preds)):
#             nslice, r = rois[i]
#             file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (filename.split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],y[i]))
#
#             # if preds[i]>0.8:
#             #     logging.info("++ Good candidate found with (nslice,x,y,diam,score): %d,%d,%d,%.2f,%.2f" % (nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i]))
#
#         logging.info("Total ROIS:%d, Good Candidates:%d, Time processnig:%.2f" % (len(preds), len([p for p in preds if p>0.5
#         ]), time()-tstart))




## MULTI PARALLEL ---------------------------------------------------------------------------------------------


def worker(filename, q):
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, thickness=1)
    logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
    q.put((filename,X,y,rois))

def listener(q):
    '''listens for messages on the q, writes to file. '''

    f = open(OUTPUT_CSV, 'wb')
    while 1:
        m = q.get()
        if m == 'kill':
            logging.info('[LISTENER] Received kill!')
            f.write('killed')
            break

        filename, x, y, rois = m
        logging.info("[LISTENER] Predicting patient %s" % (filename.split('/')[-1]))
        xf, yf, ref_filenames, roisf = [], [], [], []
        for i in range(len(x)):
            ref_filenames.extend([filename]*len(x[i]))
            xf.extend(x[i])
            yf.extend(y[i])
            roisf.extend(rois[i])

        logging.info("++ patient %s with %d ones" % (filename.split('/')[-1], np.sum(yf)))
        xf = np.asarray(xf)
        preds = model.predict(xf, verbose=1)
        logging.info("[LISTENER] Predicted patient %s, storing results" % (filename.split('/')[-1]))
        logging.info("[LISTENER] Batch results: %d/%d (th=0.7)" % (len([p for p in preds if p>0.7]),len(preds)))
        for i in range(len(preds)):
            nslice, r = roisf[i]
            f.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (ref_filenames[i].split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],yf[i]))
        f.flush()
    f.close()


def main():
    #must use Manager queue here, or will not work
    manager = multiprocessing.Manager()
    q = manager.Queue()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for filename in file_list[0:10]:
        job = pool.apply_async(worker, (filename, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()


    pool.join()
    #now we are done, kill the listener
    logging.info('Sending kill...')
    q.put('kill')
    pool.close()


if __name__ == "__main__":
   main()