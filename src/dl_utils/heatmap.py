import numpy as np
from skimage import measure


def extract_regions_from_heatmap(heatmap, threshold=None):
    """
        Given a heatmap and a threshold in it (float),
        extracts measure.regions from it
    """
    if threshold is None:
        threshold = 0.8*np.max(heatmap)  # np.mean(nodule_mask)
    thr = np.where(heatmap < threshold, 0., 1.0)  # threshold detected regions
    label_image = measure.label(thr)  # label them
    labels = label_image.astype(int)
    regions = measure.regionprops(labels, heatmap)
    return regions
