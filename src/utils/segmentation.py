import numpy as np
import scipy
import plotting
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize


def segment_lungs(image, fill_lung=True, method='Thresholding'):

    if method == 'Thresholding':
        binary_image = __segment_by_thresholding__(image, fill_lung_structures=fill_lung)
    elif method == 'KMeans':
        raise NotImplementedError("Segmentation based on KMeans not implemented.")
    elif method == 'Otsu':
        raise NotImplementedError("Segmentation based on Otsu thresholding not implemented.")
    else:
        raise NotImplementedError("Segmentation method not implemented.")

    return binary_image


def __segment_by_thresholding__(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = __largest_label_volume__(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside  body
    labels = measure.label(binary_image, background=0)
    l_max = __largest_label_volume__(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    # Binary dilation
    binary_image = __dilate__(binary_image, 3)

    return binary_image


def __largest_label_volume__(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def __dilate__(image, iterations_dilate):

    if iterations_dilate < 1:
        iterations_dilate = 1

    dilated_image = scipy.ndimage.morphology.binary_dilation(image, iterations=iterations_dilate)

    return dilated_image


def luna_segmentation(img):
    """Perform segmentation as defined in LUNA code."""
    img = (img-np.mean(img))/np.std(img)  # Standardize the pixel values
    middle = img[100:400,100:400]  # Find the average pixel value near the lungs to renormalize washed out images
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    
    # Using K-means to separate foreground (radio-opaque tissue) and background (radio transparent tissue ie lungs)
    # Doing this only on the center of the image to avoid the non-tissue parts of the image as much as possible
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
    
    # I found an initial erosion helpful for removing graininess from some of the regions and then large dilation is
    # used to make the lung region engulf the vessels and incursions into the lung cavity by radio opaque tissue
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    
    # Label each region and obtain the region properties. The background region is removed by removing regions 
    # with a bbox that is to large in either dimension. Also, the lungs are generally far away from the top
    # and bottom of the image, so any regions that are too close to the top and bottom are removed
    # This does not produce a perfect segmentation of the lungs from the image, but it is surprisingly good considering
    # its simplicity.
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.zeros((512, 512), dtype=np.int8)
    
    # The mask here is the mask for the lungs--not the nodes
    # After just the lungs are left, we do another large dilation
    # in order to fill in and out the lung mask 
    for N in good_labels:
        mask += np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10])) # one last dilation
    
    return mask


def luna_apply_mask(img, mask):
    """Given a segmentation mask, apply to the img (isolate lungs)."""
    
    new_size = [512,512]   # we're scaling back up to the original size of the image
    img = mask*img          # apply lung mask
    
    # renormalizing the masked image (in the mask region)
    new_mean = np.mean(img[mask>0])  
    new_std = np.std(img[mask>0])
    
    # pulling the background color up to the lower end of the pixel range for the lungs
    old_min = np.min(img)  # background color
    img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
    img -= new_mean
    img /= new_std
    
    # make image bounding box  (min row, min col, max row, max col)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    
    # finding the global min and max row over all regions
    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]: min_row = B[0]
        if min_col > B[1]: min_col = B[1]
        if max_row < B[2]: max_row = B[2]
        if max_col < B[3]: max_col = B[3]
    width = max_col - min_col
    height = max_row - min_row
    if width > height:
        max_row=min_row+width
    else:
        max_col = min_col+height
        
    # cropping the image down to the bounding box for all regions
    # (there's probably an skimage command that can do this in one line)
    img = img[min_row:max_row,min_col:max_col]
    mask =  mask[min_row:max_row,min_col:max_col]
    if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
        return
    else:
        # moving range to -1 to 1 to accommodate the resize function
        img = (img - np.mean(img))/(np.max(img) - np.min(img))
        new_img = resize(img, [512,512])
        return new_img


