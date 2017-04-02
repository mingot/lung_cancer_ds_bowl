import numpy as np
import logging
import multiprocessing
from time import time
from skimage import measure, transform, morphology
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    # hard coded normalization as in https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def get_regions(mask, threshold=None):
    if threshold is None:
        threshold = np.mean(mask)

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
    def __init__(self, region=None, bbox=None):
        if region is not None:
            self.bbox = region.bbox
            self.centroid = region.centroid
            self.equivalent_diameter = region.equivalent_diameter
        elif bbox is not None:
            self.bbox = bbox

    def augment_region(self, margin=5):
        self.bbox = [max(self.bbox[0]-margin,0), max(self.bbox[1]-margin,0), self.bbox[2]+margin, self.bbox[3]+margin]

    def cropped_image(self, img, output_size=(40,40)):
        x1,y1,x2,y2 = self.bbox
        if len(img.shape)==2:  # a single image to be cropped
            cropped = img[x1:x2,y1:y2]
            return transform.resize(cropped, output_size)

        elif len(img.shape)==3:  #
            stack_crops = []
            cropped = img[:,x1:x2,y1:y2]
            for nslice in range(img.shape[0]):
                stack_crops.append(transform.resize(cropped[nslice], output_size))
            return np.stack(stack_crops)


def calc_area(r):
    return (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1])


def calc_ratio(r):
    return (r.bbox[2]-r.bbox[0])*1.0/(r.bbox[3]-r.bbox[1])


def augment_bbox(r, margin=5):
    """Increase pixels by margin."""
    r.bbox = (max(r.bbox[0]-margin,0), max(r.bbox[1]-margin,0), r.bbox[2]+margin, r.bbox[3]+margin)
    return r


def extract_rois_from_lung_mask(lung_image, lung_mask, margin=5):
    """
        Given a lung image,  generate ROIs based on HU filtering.
        Reduce the candidates by discarding smalls and very rectangular regions.
    """
    mask = lung_image.copy()
    mask[lung_mask!=1] = -2000
    mask[mask<-500] = -2000  # based on LUNA examination ()

    # generate regions
    # mask = morphology.opening(mask)
    regions_pred = get_regions(mask, threshold=np.mean(mask))

    # discard small regions or long connected regions
    sel_regions = []
    for region in regions_pred:
        area, ratio = calc_area(region), calc_ratio(region)
        if 3*3<=area: #and area<=70*70 and 1.0/3<=ratio and ratio<=3:  # regions in [2.1mm, 40mm]
            sel_regions.append(region)
    regions_pred = sel_regions


    # increase the padding of the regions by 5px
    regions_pred_augmented = []
    for region in regions_pred:
        region = AuxRegion(region)
        region = augment_bbox(region, margin=margin)
        regions_pred_augmented.append(region)

    return regions_pred_augmented


def extract_crops_from_regions(img, regions, output_size=(40,40)):
    # Crop img given a vector of regions.
    # If img have depth (1 dim of 3), generate the depth cropped image
    cropped_images = []
    for region in regions:
        x1,y1,x2,y2 = region.bbox
        if len(img.shape)==2:  # a single image to be cropped
            cropped = img[x1:x2,y1:y2]
            cropped_images.append(transform.resize(cropped, output_size))

        elif len(img.shape)==3:  #
            stack_crops = []
            cropped = img[:,x1:x2,y1:y2]
            for nslice in range(img.shape[0]):
                stack_crops.append(transform.resize(cropped[nslice], output_size, mode="reflect"))
            cropped_images.append(np.stack(stack_crops))


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

    stats['tp'] = len(regions_real) - stats['fn'] # int(np.sum(labels))
    stats['fp'] = len(labels) - np.sum(labels)
    return labels, stats


def extract_rois_from_df(nodules_df):
    regions = []
    for idx, row in nodules_df.iterrows():
        x, y, d = int(row['x']), int(row['y']), int(row['diameter']+10)
        a = AuxRegion(bbox = [max(0,x-d/2), max(0,y-d/2), x+d/2, y+d/2])
        regions.append(a)
    return regions


def add_stats(stat1, stat2):
    stat_res = {}
    stat_master = stat1 if len(stat1)>0 else stat2
    for k in stat_master:
        stat_res[k] = stat1.get(k,0) + stat2.get(k,0)

    return stat_res



def load_patient(patient_data, patient_nodules_df=None, discard_empty_nodules=False,
                 output_rois=False, debug=False, include_ground_truth=False, thickness=0):
    """
    Returns images generated for each patient.
     - patient_nodules_df: pd dataframe with at least: x, y, nslice, diameter
     - thickness: number of slices up and down to be taken
     - include_ground_truth: should the ground truth patches be included as regions?
    """
    X, Y, rois = [], [], []
    total_stats = {}

    # load the slices to swipe
    if patient_nodules_df is not None:
        nslices = list(set(patient_nodules_df['nslice']))
    else:
        nslices = range(patient_data.shape[1])

    # Check if it has nodules annotated
    if patient_data.shape[0]!=3:
        aux = np.zeros((3,patient_data.shape[1], patient_data.shape[2], patient_data.shape[3]))
        aux[0] = patient_data[0]
        aux[1] = patient_data[1]
        patient_data = aux


    for nslice in nslices:
        lung_image, lung_mask, nodules_mask = patient_data[0,nslice,:,:], patient_data[1,nslice,:,:], patient_data[2,nslice,:,:]

        if patient_nodules_df is None:
            # Discard if no nodules
            if nodules_mask.sum() == 0 and discard_empty_nodules:
                continue

            # Discard if bad segmentation
            voxel_volume_l = 2*0.7*0.7/(1000000.0)
            lung_volume_l = np.sum(lung_mask)*voxel_volume_l
            if lung_volume_l < 0.009 or lung_volume_l > 0.1:
                continue  # skip slices with bad lung segmentation

            # Filter ROIs to discard small and connected
            regions_pred = extract_rois_from_lung_mask(lung_image, lung_mask)
            if include_ground_truth: regions_pred.extend(get_regions(nodules_mask,threshold=np.mean(nodules_mask)))

        else:
            sel_patient_nodules_df = patient_nodules_df[patient_nodules_df['nslice']==nslice]
            regions_pred = extract_rois_from_df(sel_patient_nodules_df)

        # Generate labels
        if np.sum(nodules_mask)!=0:
            regions_real = get_regions(nodules_mask, threshold=np.mean(nodules_mask))
            labels, stats = get_labels_from_regions(regions_real, regions_pred)
        else:
            stats = {'fp':len(regions_pred), 'tp': 0, 'fn':0}
            labels = [0]*len(regions_pred)

        # Extract cropped images
        if thickness>0:  # add extra images as channels for thick resnet
            lung_image = patient_data[0,(nslice - thickness):(nslice + thickness + 1),:,:]
            if lung_image.shape[0] != 2*thickness + 1:  # skip the extremes
                continue
        cropped_images = extract_crops_from_regions(lung_image, regions_pred)


        total_stats = add_stats(stats, total_stats)
        if debug: logging.info("++ Slice %d, stats: %s" % (nslice, str(stats)))

        X.extend(cropped_images)
        Y.extend(labels)  # nodules_mask
        rois.extend([(nslice, r) for r in regions_pred])  # extend regions with the slice index

    return (X, Y, rois, total_stats) if output_rois else (X, Y)



def multiproc_crop_generator(filenames, out_x_filename, out_y_filename, load_patient_func, parallel=False, store=True):
    """loads patches in parallel and stores the results."""

    total_stats = {}
    xf, yf = [], []
    tstart = time()
    if parallel:
        pool =  multiprocessing.Pool(4)
        tstart = time()
        x, y, stats = zip(*pool.map(load_patient_func, filenames))

        for i in range(len(x)):
            xf.extend(x[i])
            yf.extend(y[i])
            total_stats = add_stats(total_stats, stats[i])
        pool.close()
        pool.join()
    else:
        for idx,filename in enumerate(filenames):
            logging.info("Loading %d/%d" % (idx,len(filenames)))
            x,y,stats = load_patient_func(filename)
            xf.extend(x)
            yf.extend(y)
            total_stats = add_stats(total_stats, stats)


    logging.info('Total time: %.2f, total patients:%d, stats: %s' % (time() - tstart, len(x), total_stats))
    if store:
        np.savez_compressed(out_x_filename, np.asarray(xf))
        np.savez_compressed(out_y_filename, np.asarray(yf))
        logging.info('Finished saving files')