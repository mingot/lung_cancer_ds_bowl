import numpy as np
from skimage import measure, transform



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


def extract_rois_from_lungs(lung_image, lung_mask):
    """
        Given a lung image,  generate ROIs based on HU filtering.
        Reduce the candidates by discarding smalls and very rectangular regions.
    """
    mask = lung_image.copy()
    mask[lung_mask!=1] = -2000
    mask[mask<-500] = -2000  # based on LUNA examination ()

    # generate regions
    regions_pred = get_regions(mask, threshold=np.mean(mask))

    # discard small regions or long connected regions
    sel_regions = []
    for region in regions_pred:
        area, ratio = calc_area(region), calc_ratio(region)
        if 3*3<=area and area<=55*55 and 1.0/3<=ratio and ratio<=3:  # regions in [2.1mm, 40mm]
            sel_regions.append(region)
    regions_pred = sel_regions


    # increase the padding of the regions by 5px
    regions_pred_augmented = []
    for region in regions_pred:
        region = AuxRegion(region)
        region = augment_bbox(region, margin=5)
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

    stats['tp'] = np.sum(labels)
    stats['fp'] = len(labels) - np.sum(labels)
    return labels, stats

