import itertools

import numpy as np
import scipy
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing, dilation, erosion
from skimage.segmentation import clear_border
from skimage.transform import resize
from sklearn.cluster import KMeans
from skimage import measure, morphology, segmentation
import scipy.ndimage as ndimage


def segment_lungs(image, fill_lung=True, method='thresholding'):

    if method == 'thresholding1':
        binary_image = __segment_by_thresholding__(image, fill_lung_structures=fill_lung)
    elif method == 'thresholding2':
        binary_image = __segment_by_thresholding_2__(image)
    elif method == 'watershed':
        binary_image = __segment_by_watershed(image)
    else:
        raise NotImplementedError("Segmentation method not implemented.")

    return binary_image


## Checks
# # patient_file = "/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/luna0123/1.3.6.1.4.1.14519.5.2.1.6279.6001.303421828981831854739626597495.mhd"
# image_wrong = pix_resampled.copy()
# binary_image = np.array(image_wrong > -320, dtype=np.int8) + 1
#
# # patient_file = "/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/luna/luna0123/1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd"
# image = pix_resampled.copy()
# binary_image = np.array(image > -320, dtype=np.int8) + 1
#
# b = erosion(clear_border(binary_image[70]))
# labels = label(b)
# plt.imshow(b)
# plt.imshow(binary_image[70])
# plt.imshow(labels)
# plt.show()


# SEGMENTATION 1 : ---------------------------------------------------------------------------------------------------

def __segment_by_thresholding__(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8) + 1
    # binary_image = __clear_border_stack(binary_image)
    labels = label(dilation(binary_image))  # dilate the image to avoid gaps in conex components
    #labels = label(dilation(dilation(erosion(binary_image))))  # version 1
    # labels = label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    extra_points = [(0, 3, 3), (0, 3, -3), (0, -3, 3), (0, -3, -3),
                    (-1, 3, 3), (-1, 3, -3), (-1, -3, 3), (-1, -3, -3)]

    background_labels = set()
    for corner in extra_points:
        background_labels.add(labels[corner])
    for z, x, y in itertools.product([-1, 0], repeat=3):
        background_labels.add(labels[z, x, y])
    n_z, n_x, n_y = image.shape
    background_labels |= set(np.unique(labels[0:n_z, 0:n_x, 0]))
    background_labels |= set(np.unique(labels[0:n_z, 0, 0:n_y]))

    # change value of background labels
    for l in background_labels:
        binary_image[labels == l] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = label(axial_slice)
            l_max = __largest_label_volume__(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside  body
    labels = label(binary_image, background=0)
    l_max = __largest_label_volume__(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    # Morphological 3D closing
    ball_11 = __make_ball_structuring_element__(5, spatial_scaling=[2, 0.7, 0.7])
    ball_7 = __make_ball_structuring_element__(5, spatial_scaling=[2, 0.7, 0.7])
    binary_image = ndi.morphology.binary_dilation(binary_image, structure=ball_11, iterations=2)
    binary_image = ndi.morphology.binary_erosion(binary_image, structure=ball_7, iterations=2)

    return binary_image


def __make_ball_structuring_element__(rad, spatial_scaling=[1, 1, 1]):
    """
    Creates a ball of radius R, centered in the coordinates center
    @param rad: radius of the ball, in mm
    @param center: center of the ball (slice, x, y) in pixel coordinates
    @spatialSpacing

    returns a list of coordinates (x, y, z) that are in the ball of radius r centered in 0.
    """
    div = [rad / spatial_scaling[0], rad / spatial_scaling[1], rad / spatial_scaling[2]]
    # Generate the mesh of candidates
    r = np.ceil(div).astype(int)  # anisotropic spacing
    x, y, z = np.meshgrid(range(-r[0], r[0] + 1), range(-r[1], r[1] + 1), range(-r[2], r[2] + 1))
    mask = (x*spatial_scaling[0])**2 + (y * spatial_scaling[1])**2 + (z * spatial_scaling[2])**2 <= rad**2
    return mask


def __largest_label_volume__(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def __clear_border_stack(im_cube):
    result = []
    for i in range(im_cube.shape[0]):
        result.append(clear_border(im_cube[i]))

    return np.stack(result)



# SEGMENTATION 2 : ---------------------------------------------------------------------------------------------------

def __segment_by_thresholding_2__(image):

    binary_image = np.zeros(shape=image.shape)
    for k in range(0, image.shape[0]):
        binary_image[k, :, :] = __segment_lung_2d_image__(image[k, :, :])

    eroded_image = ndi.morphology.binary_erosion(binary_image, iterations=3)
    final_mask = ndi.morphology.binary_dilation(eroded_image, iterations=3)

    return final_mask


def __segment_lung_2d_image__(im):
    """
    This function segments the lungs from the given 2D slice.
    """
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < -400
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)

    background_labels = set()
    n_x, n_y = label_image.shape
    background_labels |= set(np.unique(label_image[0:n_x, 0]))
    background_labels |= set(np.unique(label_image[0:n_x, -1]))
    background_labels |= set(np.unique(label_image[0, 0:n_y]))
    background_labels |= set(np.unique(label_image[-1, 0:n_y]))

    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()

    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.label in background_labels:
                label_image[coordinates[0], coordinates[1]] = 0
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    # change value of background labels
    #    for l in background_labels:
    #        binary[label_image == l] = False

    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    return binary




# SEGMENTATION 3 : improved-lung-segmentation-using-watershed --------------------------------------------------------
# Kernel link: https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed


def __segment_by_watershed(image):
    binary_image = np.zeros(shape=image.shape)
    for k in range(0, image.shape[0]):
        binary_image[k, :, :] = __seperate_lungs(image[k, :, :])

    return binary_image

# Some of the starting Code is taken from ArnavJain, since it's more readable then my own
def __generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros(image.shape, dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def __seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = __generate_markers(image)

    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)

    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones(image.shape))

    return lungfilter


# tstart = time()
# test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, \
# test_marker_internal, test_marker_external, test_marker_watershed = _seperate_lungs(image_wrong[67])
# print time()-tstart
#
# plt.imshow(image_wrong[65])
# plt.imshow(test_lungfilter)
# plt.show()

# tstart = time()
# lung_mask = segment_lungs(image_wrong, method="Thresholding2")
# print time()-tstart
#
# plt.imshow(lung_mask[70])
# plt.show()
#
# plotting.cube_show_slider(lung_mask)