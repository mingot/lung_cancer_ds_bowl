import os
import random
import logging
import numpy as np
from time import time
import matplotlib.pyplot as plt
from utils import plotting
from skimage import measure
from skimage import transform
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard



def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    # hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def get_regions(mask, threshold=None):
    if threshold is None:
        threshold = np.mean(mask)  # np.mean(mask)
    thr = np.where(mask < threshold, 0., 1.0)
    label_image = measure.label(thr)  # label them
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    return regions


def intersection_regions(r1, r2):
    h = min(r1.bbox[2], r2.bbox[2]) - max(r1.bbox[0], r2.bbox[0])
    w = min(r1.bbox[3], r2.bbox[3]) - max(r1.bbox[1], r2.bbox[1])
    intersectionArea = w*h
    if h<0 or w<0:
        return 0.0

    area1 = (r1.bbox[2] - r1.bbox[0])*(r1.bbox[3] - r1.bbox[1])
    area2 = (r2.bbox[2] - r2.bbox[0])*(r2.bbox[3] - r2.bbox[1])
    unionArea = area1 + area2 - intersectionArea
    overlapArea = intersectionArea*1.0/unionArea # This should be greater than 0.5 to consider it as a valid detection.
    return overlapArea


class AuxRegion():
    """Auxiliar class to change the bbox of regions props."""
    def __init__(self, region):
        self.bbox = region.bbox
        self.centroid = region.centroid
        self.equivalent_diameter = region.equivalent_diameter


def calc_area(r):
    return (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1])


def calc_ratio(r):
    return (r.bbox[2]-r.bbox[0])*1.0/(r.bbox[3]-r.bbox[1])


def augment_bbox(r, margin=5):
    """Increase pixels by margin."""
    r.bbox = (r.bbox[0]-margin, r.bbox[1]-margin, r.bbox[2]+margin, r.bbox[3]+margin)
    return r


def extract_rois_from_lungs(lung_image, lung_mask):
    """
        Given a lung image,  generate ROIs based on HU filtering.
        Reduce the candidates by discarding smalls and very rectangular regions.
    """
    mask = lung_image.copy()
    mask[lung_mask!=1] = -2000
    mask[mask<-500] = -2000  # based on LUNA examination ()

    regions_pred = get_regions(mask, threshold=np.mean(mask))
    # plotting.plot_bb(mask, regions_pred)

    # discard small regions or long connected regions
    for region in regions_pred[:]:
        if calc_area(region)<3*3 or calc_area(region)>55*55:  # regions in [2.1mm, 40mm]
            regions_pred.remove(region)
        elif calc_ratio(region)>3 or calc_ratio(region)<1.0/3:
            regions_pred.remove(region)

    # increase the padding of the regions by 5px
    regions_pred_augmented = []
    for region in regions_pred:
        region = AuxRegion(region)
        region = augment_bbox(region, margin=5)
        regions_pred_augmented.append(region)

    return regions_pred_augmented


def extract_crops_from_regions(img, regions, output_size=(40,40)):
    # Crop images
    cropped_images = []
    for region in regions:
        #zeros = np.zeros((40,40))
        # region = regions_pred[3]
        x1,y1,x2,y2 = region.bbox
        cropped = img[x1:x2,y1:y2]
        # zeros.fill(-500)
        # xini = 40/2-(x2-x1)/2
        # yini = 40/2-(y2-y1)/2
        # h = x2 - x1
        # w = y2 - y1
        # zeros[xini:(xini+h), yini:(yini+w)] = cropped
        cropped_images.append(transform.resize(cropped, output_size))
    return cropped_images


def get_labels_from_regions(regions_real, regions_pred):
    """Extract labels (0/1) from regions."""
    labels = [0]*len(regions_pred)
    stats = {'fn': 0, 'tp': 0, 'fp': 0}
    for region_real in regions_real:
        is_detected = False
        for idx,region_pred in enumerate(regions_pred):
            # discard regions that occupy everything
            if region_real.bbox[0]==0 or region_pred.bbox[0]==0:
                continue
            score = intersection_regions(r1=region_pred, r2=region_real)
            if score>.1:
                labels[idx] = 1
                is_detected = True
        if not is_detected:
            stats['fn'] += 1

    stats['tp'] = np.sum(labels)
    stats['fp'] = len(labels) - np.sum(labels)
    return labels, stats



### LOADING DATA -----------------------------------------------------------------

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=.06,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.0002,
    zoom_range=0.0002,
    dim_ordering="th",
    horizontal_flip=True
    )

def load_patient(filename, discard_empty_nodules=True, output_rois=False):

    X, Y, rois = [], [], []
    logging.info('Loading patient %s' % filename)

    # filename = 'luna_121805476976020513950614465787.npz'
    # j = 46
    t_start = time()
    b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']

    # Check if it has nodules annotated
    if b.shape[0]!=3:
        if discard_empty_nodules:
            return X, Y  # return empty
        else:  # create a dummy zero nodules mask (for testing)
            aux = np.zeros((3,b.shape[1], b.shape[2], b.shape[3]))
            aux[0] = b[0]
            aux[1] = b[1]
            b = aux

    last_slice = -1e3  # big initialization
    slices = []
    for j in range(b.shape[1]):

        lung_image, lung_mask, nodules_mask = b[0,j,:,:], b[1,j,:,:], b[2,j,:,:]

        # Discard if no nodules
        if nodules_mask.sum() == 0 and discard_empty_nodules:
            continue

        # Discard if bad segmentation
        voxel_volume_l = 2*0.7*0.7/(1000000.0)
        lung_volume_l = np.sum(lung_mask)*voxel_volume_l
        if lung_volume_l < 0.02 or lung_volume_l > 0.1:
            continue  # skip slices with bad lung segmentation

        # # Discard if consecutive slices
        # if j<last_slice + 5:
        #     continue

        # Filter ROIs to discard small and connected
        regions_pred = extract_rois_from_lungs(lung_image, lung_mask)
        regions_real = get_regions(nodules_mask, threshold=np.mean(nodules_mask))

        ## visualize regions
        # plotting.plot_bb(lung_image, regions_real)
        # plotting.plot_bb(lung_image, regions_pred)

        # Extract cropped images
        cropped_images = extract_crops_from_regions(lung_image, regions_pred)

        # Generate labels
        labels, stats = get_labels_from_regions(regions_real, regions_pred)
        logging.info('ROIs stats for slice %d: %s' % (j, str(stats)))

        # if ok append
        last_slice = j
        slices.append(j)

        X.extend(cropped_images)
        Y.extend(labels)  # nodules_mask
        rois.extend([(j,r) for r in regions_pred])  # extend regions with the slice index

    logging.info('Finished in %.2f s! Added %d slices: %s' % (time()-t_start, len(slices), str(slices)))

    return (X, Y, rois) if output_rois else (X, Y)


def chunks(file_list=[], batch_size=32, augmentation_times=4):

    CONCURRENT_PATIENTS = 10  # Load more than 1 patient at a time to have diversity
    while True:
        for j in range(0,len(file_list),CONCURRENT_PATIENTS):
            filenames = file_list[j:(j+CONCURRENT_PATIENTS)]
            X, y = [], []
            for filename in filenames:
                X_single, y_single = load_patient(filename)
                if len(X_single)==0:
                    continue
                X.extend(X_single)
                y.extend(y_single)

            # downsample negatives (reduce 90%)
            selected_samples  = [i for i in range(len(y)) if y[i]==1 or random.randint(0,9)==0]
            X = [X[i] for i in selected_samples]
            y = [y[i] for i in selected_samples]
            logging.info("Final downsampled dataset stats: TP:%d, FP:%d" % (sum(y), len(y)-sum(y)))

            # convert to np array and add extra axis (needed for keras)
            X = np.expand_dims(np.asarray(X),axis=1)
            y = np.expand_dims(np.asarray(y),axis=1)

            i = 0
            for X_batch, y_batch in datagen.flow(X, y, batch_size=batch_size, shuffle=True):
                i += 1
                if i>len(X)*augmentation_times:  # stop when we have augmented enough the batch
                    break
                if X_batch.shape[0]!=batch_size:  # ensure correct batch size
                    continue
                yield X_batch, y_batch




### MODEL LOASING -----------------------------------------------------------------


# PARAMETERS
PATIENTS_VALIDATION = 20  # number of patients to validate the model on
USE_EXISTING = True  # load previous model to continue training or test


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
OUTPUT_MODEL = wp + 'models/jm_patches_train_v04.hdf5'
OUTPUT_CSV = wp + 'output/noduls_patches_v04_dsb.csv'
LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard, model checkpoint and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

# # Visualize a batch
# X_batch, y_batch  = chunks(file_list_train, batch_size=32).next()
# plotting.multiplot([X_batch[i,0] for i in range(X_batch.shape[0])],
#                    [y_batch[i,0] for i in range(y_batch.shape[0])])


# Load model
model = ResnetBuilder().build_resnet_50((1,40,40),1)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])
if USE_EXISTING:
    print 'Loading exiting model...'
    model.load_weights(OUTPUT_MODEL)
    #model.load_weights(wp + 'models/jm_patches_train_v3.hdf5')


### TRAINING -----------------------------------------------------------------

# # PATIENTS FILE LIST
# file_list = os.listdir(INPUT_PATH)
# file_list = [g for g in file_list if g.startswith('luna_')]
# random.shuffle(file_list)
# file_list_test = file_list[-PATIENTS_VALIDATION:]
# file_list_train = file_list[:-PATIENTS_VALIDATION]
# logging.info("Test patients: %s" % str(file_list_test))
#
#
# model.fit_generator(generator=chunks(file_list_train, batch_size=32),
#                     samples_per_epoch=1280,  # make it small to update TB and CHECKPOINT frequently
#                     nb_epoch=500,
#                     verbose=1,
#                     callbacks=[tb, model_checkpoint],
#                     validation_data=chunks(file_list_test, batch_size=32),
#                     nb_val_samples=32*20,
#                     max_q_size=64,
#                     nb_worker=1)  # a locker is needed if increased the number of parallel workers

# while True:
#     try:
#         a = chunks(file_list_train, batch_size=32).next()
#         X, Y = a
#     except:
#         print "Error!! try catch"
#         print a
#     if X is None:
#         print "Error!! NONE FOUND!!"
#         break


### TESTING -----------------------------------------------------------------



PREDICTION_THRESHOLD = .1
file_list = os.listdir(INPUT_PATH)
file_list = [g for g in file_list if g.startswith('dsb_')]


with open(OUTPUT_CSV, 'w') as file:

    # write the header
    file.write('filename,nslice,x,y,diameter,score\n')

    for idx, filename in enumerate(file_list):
        logging.info("Patient %s (%d/%d)" % (filename, idx, len(file_list)))
        #filename = file_list[2]
        # b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
        X, y, rois = load_patient(filename, discard_empty_nodules=False, output_rois=True)
        #plotting.multiplot(X[0:15])

        X = np.expand_dims(np.asarray(X),axis=1)
        preds = model.predict(X, verbose=1)

        for i in range(len(preds)):
            #if preds[i]>PREDICTION_THRESHOLD:
            nslice, r = rois[i]
            #print '%s,%d,%d,%d,%.3f\n' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter)
            file.write('%s,%d,%d,%d,%.3f,%.5f\n' % (filename,nslice,r.centroid[0], r.centroid[1], r.equivalent_diameter,preds[i]))

        np.mean(preds)


# ## Checking
# b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
# for j in range(b.shape[1]):
#     if np.sum(b[2,j])!=0:
#         print j
# plotting.plot_mask(b[0,96], b[2,96])



# ## Calculate area regions of luna
# for idx, filename in enumerate(file_list):
#     b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
#     if b.shape[0]!=3:
#         continue
#
#     print 'Loading %s (%d/%d)' % (filename, idx, len(file_list))
#     for j in range(b.shape[1]):
#         if np.sum(b[2,j])!=0:
#             regions = get_regions(b[2,j])
#             for region in regions:
#                 print "Filename %s, slice %d, area %s" % (filename, j, str(calc_area(region)))


