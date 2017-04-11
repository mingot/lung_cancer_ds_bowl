import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from time import time
from dl_model_patches import  common



# ## if the OUTPUT_CSV file already exists, continue it
# previous_filenames = set()
# if os.path.exists(OUTPUT_CSV):
#     write_method = 'a'
#     with open(OUTPUT_CSV) as file:
#         for l in file:
#             previous_filenames.add(l.split(',')[0])


# ## Loading DF (if necessary)
# OUTPUT_DL1 = wp + 'output/nodules_patches_dl1_v11.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
# OUTPUT_DL2 = wp + 'output/nodules_patches_hardnegative_v03.csv'  # OUTPUT_DL1 = wp + 'personal/noduls_patches_v06.csv'
#
# logging.info("Loading DL1 and DL2 data frames...")
# dl1_df = pd.read_csv(OUTPUT_DL1)
# dl1_df = dl1_df[dl1_df['patientid'].str.startswith('dsb')]  # Filter DSB patients
# dl2_df = pd.read_csv(OUTPUT_DL2)
# merge_df = pd.merge(dl1_df, dl2_df, on=['patientid','nslice','x','y','diameter'], how='inner', suffixes=('_dl1', '_dl2'))
# nodules_df = merge_df[((merge_df['score_dl1'] + merge_df['score_dl2'])/2 > 0.5) & (merge_df['diameter']>7)]   # 12k candidates



## MULTI PARALLEL ---------------------------------------------------------------------------------------------
import multiprocessing


def worker(filename, q, nodules_df=None):
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

def listener(q, model_path, output_csv):
    """Reads regions from queue, predicts nodules and stores in the output file."""
    from keras import backend as K
    from dl_networks.sample_resnet import ResnetBuilder
    from keras.optimizers import Adam

    # Model loading inside the listener thread (otherwise keras complains)
    K.set_image_dim_ordering('th')
    model = ResnetBuilder().build_resnet_50((3,40,40),1)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
    logging.info('Loading existing model %s...' % model_path)
    model.load_weights(model_path)

    total, errors = 0, 0

    f = open(output_csv, 'w')
    f.write('patientid,nslice,x,y,diameter,score,label\n')
    while 1:
        queue_element = q.get()
        if queue_element == 'kill':
            logging.info('[LISTENER] Closing...')
            break

        try:
            filename, x, y, rois = queue_element
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


def evaluate_model(file_list, model_path, output_csv, nodules_df=None):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    pool = multiprocessing.Pool(5)  # multiprocessing.cpu_count()

    #put listener to work first
    watcher = pool.apply_async(listener, (q, model_path, output_csv))

    #fire off workers
    jobs = []
    for filename in file_list:
        job = pool.apply_async(worker, (filename, q, nodules_df))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    watcher.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    # python evaluate.py --input_path=/mnt/hd2/test_difs/ --model=~/lung_cancer_ds_bowl/models/jm_patches_train_v19.hdf5 --output_csv=~/prova_dl1_19.csv
    parser = argparse.ArgumentParser(description='Evaluates a DL model over some patients')
    parser.add_argument('--input_path', help='Path of the preprocessed patient files')
    parser.add_argument('--model', help='DL model to use')
    parser.add_argument('--output_csv', help='Output CSV with nodules and scores')
    parser.add_argument('--input_csv',  help='Preselected nodules to pass to the DL')
    args = parser.parse_args()

    # DEFAULT VALUES
    wp = os.environ['LUNG_PATH']
    INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
    stage2_df = pd.read_csv(wp + 'data/stage2_sample_submission.csv')
    stage2_ids = ['dsb_%s.npz' % pid for pid in stage2_df['id']]

    #MODEL = wp + 'models/jm_patches_train_v19.hdf5'
    #OUTPUT_CSV = wp + 'output/nodules_patches_dl1_v19_stage2.csv'
    # MODEL = wp + 'models/jm_patches_hardnegative_v04.hdf5'
    # OUTPUT_CSV = wp + 'output/nodules_patches_dl2_v04_stage2.csv'
    # MODEL = wp + 'models/jm_patches_hardnegative_v03.hdf5'
    # OUTPUT_CSV = wp + 'output/nodules_patches_dl2_v03_stage2_total.csv'
    # MODEL = wp + 'models/jm_patches_train_v11.hdf5'
    # OUTPUT_CSV = wp + 'output/nodules_patches_dl1_v11_stage2_total.csv'
    MODEL = wp + 'models/jm_patches_dl3_v06.hdf5'
    OUTPUT_CSV = wp + 'output/nodules_patches_dl3_v06.csv'
    nodules_df = pd.read_csv('/home/mingot/dl3/dl12_test_dl3.csv')

    if args.input_path: INPUT_PATH = args.input_path
    if args.model: MODEL = args.model
    if args.output_csv: OUTPUT_CSV = args.output_csv
    #nodules_df = pd.read_csv(args.input_csv) if args.input_csv else None

    ## Params and filepaths
    file_list = [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH) if fp.startswith('dsb_')]
    logging.info("Processing %d files..." % len(file_list))

    tstart = time()
    evaluate_model(file_list, model_path=MODEL, output_csv=OUTPUT_CSV, nodules_df=nodules_df)
    print "Total time:", time() - tstart


