import os
import sys
import csv
import logging
import numpy as np
import pandas as pd
import multiprocessing
from time import time
sys.path.append('..')
#import dl_networks
from dl_networks.simplepos_resnet import simpleposResnet
from keras.optimizers import Adam
from dl_model_patches import  common
from dl_model_simplepos_patches import common
from keras import backend as K

wp = '/home/jose/lung_cancer_ds_bowl/'
INPUT_PATH = '/mnt/hd2/preprocessed5'
OUTPUT_MODEL = '/mnt/hd2/models/jc_simplepos_patches_train_v04.hdf5.backup'
OUTPUT_CSV = '/home/felix/lung_cancer_ds_bowl/output/simplepos_patches_v04.csv'

file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH) if fp.startswith('dsb_')]


def worker(filename, q):
    while 1:
        if q.qsize() < 10:
            patient_data = np.load(filename)['arr_0']
            #logging.info("[WORKER] "+str(filename)+" patient_data shape " +str(patient_data.shape)) 
            X, y, rois, stats = common.load_patient_3d_atlas(patient_data, discard_empty_nodules=False, output_rois=True, debug=False, thickness=1)
            #logging.info("[WORKER] len X " +str(len(X)))
            logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
            q.put((filename,X,y,rois))
            break

def listener(q):
    logging.info("[LISTENER] Started")
    """Reads regions from queue, predicts nodules and stores in the output file."""
    #from keras import backend as K
    #from dl_networks.sample_resnet import ResnetBuilder
    #from keras.optimizers import Adam
    # Model loading inside the listener thread (otherwise keras complains)
    try:    
        K.set_image_dim_ordering('th')
        #from dl_networks.simplepos_resnet import simpleposResnet
        model = simpleposResnet().get_posResnet((3,40,40),(5,)) # 5 == N_extra_features
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
        model.load_weights(OUTPUT_MODEL)
        logging.info('Loading existing model...')

        total, errors = 0, 0
        csvfile = open(OUTPUT_CSV, 'w')
        writer = csv.DictWriter(csvfile, fieldnames=['patientid','min','mean','max','std'])
        writer.writeheader()
        while 1:
            m = q.get()
            if m == 'kill':
                logging.info('[LISTENER] Closing...')
                break

            try:
                filename, X, _, _ = m
                filename = filename.split('/')[-1]
                if 0 == len(X): continue ##
                #print x #np.asarray(x).shape
                a = np.array([X[i][0] for i in range(len(X))])
                b = np.array([X[i][1] for i in range(len(X))])
                #x_numpy = np.asarray(x)
                preds = model.predict([a,b], verbose=0)
                logging.info("[LISTENER] Predicted patient %d %s" % (total, filename))
                writer.writerow({'patientid':filename,'min':preds.min(),'max':preds.max(),'std':preds.std(),'mean':preds.mean()})
                total += 1
            except Exception as e:
                logging.error("[LISTENER] Error processing result, skipping. %s" % str(e))
                errors += 1

        logging.info("Stats: %d patients, %d errors" % (total,errors))
        csvfile.close()
    except Exception as e:
        logging.info("[LISTENER] Exception "+str(e) )
    logging.info("[LISTENER] Closing")

def main():
    manager = multiprocessing.Manager()
    q = manager.Queue()
    pool = multiprocessing.Pool(5)  # multiprocessing.cpu_count()

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for filename in file_list:
        job = pool.apply_async(worker, (filename, q))
        jobs.append(job)
        #break

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # watcher.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    tstart = time()
    main()
    print "Total time:", time() - tstart
