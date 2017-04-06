import os
import sys
import logging
import numpy as np
import pandas as pd
from time import time
from dl_model_patches import  common

# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
OUTPUT_MODEL = wp + 'models/jm_patches_malign_v03.hdf5'
OUTPUT_CSV = wp + 'output/nodules_patches_dl3_v03.csv'


## Params and filepaths
#VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'
#NODULES_PATH = wp + 'data/luna/annotations.csv'
file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH) if fp.startswith('dsb_')]


# ## if the OUTPUT_CSV file already exists, continue it
# previous_filenames = set()
# if os.path.exists(OUTPUT_CSV):
#     write_method = 'a'
#     with open(OUTPUT_CSV) as file:
#         for l in file:
#             previous_filenames.add(l.split(',')[0])


## Loading DF (if necessary)
OUTPUT_DL1 = wp + 'output/nodules_patches_dl1_v11.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
OUTPUT_DL2 = wp + 'output/nodules_patches_hardnegative_v03.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'

logging.info("Loading DL1 and DL2 data frames...")
dl1_df = pd.read_csv(OUTPUT_DL1)
dl1_df = dl1_df[dl1_df['patientid'].str.startswith('dsb')]  # Filter DSB patients
dl2_df = pd.read_csv(OUTPUT_DL2)
merge_df = pd.merge(dl1_df, dl2_df, on=['patientid','nslice','x','y','diameter'], how='inner', suffixes=('_dl1', '_dl2'))
nodules_df = merge_df[((merge_df['score_dl1'] + merge_df['score_dl2'])/2 > 0.5) & (merge_df['diameter']>7)]   # 12k candidates


## TERMINAL ARGUMENTS ---------------------------------------------------------------------------------------------

#--input_path=%s --model=%s input_csv=%s --output_file=%s
for arg in sys.argv[1:]:
    if arg.startswith('--input_path='):
        INPUT_PATH = ''.join(arg.split('=')[1:])
    elif arg.startswith('--model='):
        OUTPUT_MODEL = ''.join(arg.split('=')[1:])
    elif arg.startswith('--output_file='):
        OUTPUT_CSV = ''.join(arg.split('=')[1:])
    elif arg.startswith('--input_csv='):
        nodules_df = ''.join(arg.split('=')[1:])
        nodules_df = pd.read_csv(nodules_df)
    else:
        print('Unknown argument {}. Ignoring.'.format(arg))


## MULTI PARALLEL ---------------------------------------------------------------------------------------------
import multiprocessing


def worker(filename, q):
    while 1:
        if q.qsize() < 10:
            patient_data = np.load(filename)['arr_0']
            if nodules_df is not None:
                ndf = nodules_df[nodules_df['patientid']==filename.split('/')[-1]]
                X, y, rois, stats = common.load_patient(patient_data, ndf, discard_empty_nodules=False, output_rois=True, thickness=1)
            else:
                X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, thickness=1)
            logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
            q.put((filename,X,y,rois))
            break

def listener(q):
    """Reads regions from queue, predicts nodules and stores in the output file."""
    from keras import backend as K
    from dl_networks.sample_resnet import ResnetBuilder
    from keras.optimizers import Adam

    # Model loading inside the listener thread (otherwise keras complains)
    K.set_image_dim_ordering('th')
    model = ResnetBuilder().build_resnet_50((3,40,40),1)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
    logging.info('Loading existing model...')
    model.load_weights(OUTPUT_MODEL)

    total, errors = 0, 0

    f = open(OUTPUT_CSV, 'w')
    f.write('patientid,nslice,x,y,diameter,score,label\n')
    while 1:
        m = q.get()
        if m == 'kill':
            logging.info('[LISTENER] Closing...')
            break

        try:
            filename, x, y, rois = m
            filename = filename.split('/')[-1]

            preds = model.predict(np.asarray(x), verbose=1)
            logging.info("[LISTENER] Predicted patient %d %s. Batch results: %d/%d (th=0.7)" % (total, filename, len([p for p in preds if p>0.7]),len(preds)))
            for i in range(len(preds)):
                nslice, r = rois[i]
                f.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (filename, nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],y[i]))
            total += 1
            f.flush()
        except Exception as e:
            logging.error("[LISTENER] Error processing result, skipping. %s" % str(e))
            errors += 1

    logging.info("Stats: %d patients, %d errors" % (total,errors))
    f.close()


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


# NON PARALLEL ------------------------------------------------------------------------------------------------------

# from keras.optimizers import Adam
# from dl_networks.sample_resnet import ResnetBuilder
#from keras import backend as K
# K.set_image_dim_ordering('th')

# # Load model
# model = ResnetBuilder().build_resnet_50((3,40,40),1)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
# logging.info('Loading existing model...')
# model.load_weights(OUTPUT_MODEL)

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

