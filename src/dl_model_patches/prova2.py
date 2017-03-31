import logging
import numpy as np
from dl_model_patches import common
import multiprocessing as mp
import time


fn = 'c:/temp/temp.txt'

def worker(arg, q):
    '''stupidly simulates long running process'''
    start = time.clock()
    s = 'this is a test'
    txt = s
    for i in xrange(200000):
        txt += s
    done = time.clock() - start
    with open(fn, 'rb') as f:
        size = len(f.read())
    res = 'Process' + str(arg), str(size), done
    q.put(res)
    return res

def workerbo(filename, q):
    patient_data = np.load(filename)['arr_0']
    X, y, rois, stats = common.load_patient(patient_data, discard_empty_nodules=False, output_rois=True, thickness=1)
    logging.info("Patient: %s, stats: %s" % (filename.split('/')[-1], stats))
    q.put((filename,X,y,rois))
    #return X, y, rois, stats

def listener(q):
    f = open(fn, 'wb')
    while 1:
        m = q.get()
        if m == 'kill':
            f.write('killed')
            break
        f.write(str(m) + '\n')
        f.flush()
    f.close()

def listenerbo(q):
    '''listens for messages on the q, writes to file. '''

    f = open(fn, 'wb')
    while 1:
        m = q.get()
        if m == 'kill':
            f.write('killed')
            break

        filename, x, y, rois = m
        xf, yf, ref_filenames, roisf = [], [], [], []
        for i in range(len(x)):
            ref_filenames.extend([filename]*len(x[i]))
            xf.extend(x[i])
            yf.extend(y[i])
            roisf.extend(rois[i])

        xf = np.asarray(xf)
        preds = model.predict(xf, verbose=1)
        logging.info("Batch results: %d/%d (th=0.7)" % (len([p for p in preds if p>0.7]),len(preds)))
        for i in range(len(preds)):
            nslice, r = roisf[i]
            file.write('%s,%d,%d,%d,%.3f,%.5f,%d\n' % (ref_filenames[i].split('/')[-1], nslice, r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i],yf[i]))
        f.flush()
    f.close()



def main():
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for filename in filenames:
        job = pool.apply_async(worker, (filename, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

if __name__ == "__main__":
   main()