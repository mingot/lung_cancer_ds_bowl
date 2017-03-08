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



# hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
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
    def __init__(self, dim):
        self.bbox = dim


def calc_area(r):
    return (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1])


def calc_ratio(r):
    return (r.bbox[2]-r.bbox[0])*1.0/(r.bbox[3]-r.bbox[1])


def augment_bbox(r, margin=5):
    """Increase pixels by margin."""
    r.bbox = (r.bbox[0]-margin, r.bbox[1]-margin, r.bbox[2]+margin, r.bbox[3]+margin)
    return r


def extract_rois_from_lungs(lung_img, lung_mask):
    """
        Given a lung image,  generate ROIs based on HU filtering.
        Reduce the candidates by discarding smalls and very rectangular regions.
    """
    mask = lung_img.copy()
    mask[lung_mask!=1] = -2000
    mask[mask<-500] = -2000  # based on LUNA examination ()

    regions_pred = get_regions(mask, threshold=np.mean(mask))

    # discard small regions or long connected regions
    for region in regions_pred[:]:
        if calc_area(region)<3*3:
            regions_pred.remove(region)
        elif calc_ratio(region)>3 or calc_ratio(region)<1.0/3:
            regions_pred.remove(region)

    # increase the padding of the regions by 5px
    regions_pred_augmented = []
    for region in regions_pred:
        region = AuxRegion(region.bbox)
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
    for region_real in regions_real:
        for idx,region_pred in enumerate(regions_pred):
            # discard regions that occupy everything
            if region_real.bbox[0]==0 or region_pred.bbox[0]==0:
                continue
            score = intersection_regions(r1=region_pred, r2=region_real)
            if score>.1:
                labels[idx] = 1
    return labels



### DEBUGGING -----------------------------------------------------------------

# tp,fp,fn = 0,0,0
# for filename in file_list:
#
#     # filename = 'luna_120842785645314664964010792308.npz'
#     # j = 60
#     b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
#     b.shape
#
#     if b.shape[0]!=3:
#         continue
#
#     for j in range(b.shape[1]):
#         if np.sum(b[2,j])!=0:
#             print '%s slice:%d' % (filename, j)
#
#             lung_image = b[0,j]
#             lung_mask = b[1,j]
#             nodules_mask = b[2,j]
#
#             regions_real = get_regions(nodules_mask, threshold=np.mean(nodules_mask))
#
#             # Filter ROIs to discard small and connected
#             mask = lung_image.copy()
#             mask[lung_mask!=1] = -2000
#             mask[mask<-500] = -2000  # based on LUNA examination ()
#
#             regions_pred = get_regions(mask, threshold=np.mean(mask))
#
#             # discard small regions or long connected regions
#             for region in regions_pred[:]:
#                 if calc_area(region)<3*3:
#                     regions_pred.remove(region)
#                 elif calc_ratio(region)>3 or calc_ratio(region)<1.0/3:
#                     regions_pred.remove(region)
#
#             # increase the padding of the regions by 5px
#             regions_pred_augmented = []
#             for region in regions_pred:
#                 region = AuxRegion(region.bbox)
#                 region = augment_bbox(region, margin=5)
#                 regions_pred_augmented.append(region)
#
#             regions_pred = regions_pred_augmented
#
#             # Extract cropped images
#             cropped_images = extract_crops_from_regions(lung_image, regions_pred)
#
#             # for idx,region in enumerate(regions_pred):
#             #     print idx, calc_area(region)
#
#             # calc_area(regions_pred[43])
#             # calc_ratio(regions_pred[43])
#             # regions_pred[43].bbox
#
#             # plt.imshow(mask)
#             # plt.show()
#             # plotting.plot_mask(mask, nodules_mask)
#             # plotting.plot_mask(lung_image, nodules_mask)
#             # plotting.plot_bb(mask, regions_pred)
#             # plotting.plot_bb(mask, regions_pred2)
#             # plotting.plot_bb(mask, [regions_pred[43]])
#             # calc_ratio(regions_pred[2])
#             # calc_area(regions_pred[1])
#             # intersection_regions(r1=regions_pred[2], r2=regions_real[0])
#
#             print '%d nodules, %d candidates' % (len(regions_real), len(regions_pred))
#
#             labels = [0]*len(regions_pred)
#             for region_real in regions_real:
#                 detected = False
#                 for idx,region_pred in enumerate(regions_pred):
#                     # discard regions that occupy everything
#                     if region_real.bbox[0]==0 or region_pred.bbox[0]==0:
#                         continue
#                     score = intersection_regions(r1=region_pred, r2=region_real)
#                     if score>0.0:
#                         print score
#                     if score>.1:
#                         tp+=1
#                         detected = True
#                         labels[idx] = 1
#                     else:
#                         fp+=1
#                 if not detected:
#                     fn += 1
#             print 'TP:%d, FP:%d, FN:%d' % (tp,fp,fn)


### LOADING DATA -----------------------------------------------------------------


datagen = ImageDataGenerator(
    rotation_range=.06,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.0002,
    zoom_range=0.0002,
    dim_ordering="th",
    horizontal_flip=True
    )



def load_patients(filelist, shuffle=False):

    X, Y = [], []
    for filename in filelist:

        b = np.load(os.path.join(INPUT_PATH, filename))['arr_0']
        if b.shape[0]!=3:
            continue

        last_slice = -1e3  # big initialization
        slices = []
        for j in range(b.shape[1]):

            lung_image, lung_mask, nodules_mask = b[0,j,:,:], b[1,j,:,:], b[2,j,:,:]

            # Discard if no nodules
            if nodules_mask.sum() == 0:
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

            # Extract cropped images
            cropped_images = extract_crops_from_regions(lung_image, regions_pred)

            # Generate labels
            labels = get_labels_from_regions(regions_real, regions_pred)

            # if ok append
            last_slice = j
            slices.append(j)

            X.extend(cropped_images)
            Y.extend(labels)  # nodules_mask

        logging.info('patient %s added %d slices: %s' % (filename, len(slices), str(slices)))

    if shuffle and len(X)>0:
        aux = list(zip(X, Y))
        random.shuffle(aux)
        X, Y = zip(*aux)

    return X, Y



def chunks(file_list=[], batch_size=32, infinite=True, augmentation_times=4):

    CONCURRENT_PATIENTS = 5  # limitation by memory
    while True:
        for j in range(0, len(file_list), CONCURRENT_PATIENTS):
            filenames = file_list[j:j+CONCURRENT_PATIENTS]
            a, b = load_patients(filenames,shuffle=True)

            # downsample negatives (reduce 90%)
            selected_samples  = [i for i in range(len(b)) if b[i]==1 or random.randint(0,9)==0]
            a = [a[i] for i in selected_samples]
            b = [b[i] for i in selected_samples]
            logging.info("TP:%d, FP:%d" % (sum(b), len(b)-sum(b)))

            # convert to np array and add extra axis (needed for keras)
            a = np.expand_dims(np.asarray(a),axis=1)
            b = np.expand_dims(np.asarray(b),axis=1)

            i = 0
            for X_batch, y_batch in datagen.flow(a, b, batch_size=batch_size):
                i += 1
                if i>len(a)*augmentation_times:
                    break
                yield X_batch, y_batch




### TRAINING -----------------------------------------------------------------

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from dl_networks.sample_resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard

# PARAMETERS
NUM_EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 15  # number of patients to test on
TRAIN = True  # should the model be trained
USE_EXISTING = False  # load previous model to continue training or test


# PATHS
wp = os.environ['LUNG_PATH']
INPUT_PATH = wp + 'data/preprocessed5_sample' # '/mnt/hd2/preprocessed5'  #
OUTPUT_MODEL = wp + 'models/jm_patches_train.hdf5'
OUTPUT_CSV = wp + 'output/AUX_noduls_unet_v03.csv'
LOGS_PATH = wp + 'logs/%s' % str(int(time()))
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# OTHER INITIALIZATIONS: tensorboard and logging
tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
K.set_image_dim_ordering('th')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# PATIENTS FILE LIST
file_list = os.listdir(INPUT_PATH)
file_list = [g for g in file_list if g.startswith('luna_')]
random.shuffle(file_list)
file_list_test = file_list[-TEST_SIZE:]
file_list_train = file_list[:-TEST_SIZE]


# X_batch, y_batch  = chunks(file_list_train,batch_size=32,infinite=True).next()
# X_batch.shape
# y_batch.shape
# for i in range(32):
#     if y_batch[i,0]==1:
#         print i
# np.sum(y_batch)
# plt.imshow(X_batch[23,0])
# plt.show()


model = ResnetBuilder().build_resnet_18((1,40,40),1)
model.compile(optimizer=Adam(lr=.5e-2), loss='binary_crossentropy', metrics=['accuracy','fmeasure'])


model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
model.fit_generator(generator=chunks(file_list_train,batch_size=32,infinite=True),
                    samples_per_epoch=300, # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=100,
                    verbose=1,
                    callbacks=[tb, model_checkpoint],
                    validation_data=chunks(file_list_test,batch_size=32,infinite=True),
                    nb_val_samples=20,  # TO REVIEW
                    max_q_size=10,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers