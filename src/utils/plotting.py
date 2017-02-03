import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def multiplot(imgs):
    """Plot multiple imags in a grid."""
    nimg = len(imgs)
    num_rows = int(math.sqrt(nimg)) + 1
    f, plots = plt.subplots(num_rows, num_rows, sharex='all', sharey='all', figsize=(num_rows, num_rows))
    for i in range(nimg):
        plots[i // num_rows, i % num_rows].axis('off')
        plots[i // num_rows, i % num_rows].imshow(imgs[i])
        #plots[i // 11, i % 11].imshow(patient_slices[i], cmap=plt.cm.bone)



def plot_bb(img, region):
    """Draw the img and the bounding box defined by a scikit image region (measure module)."""
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    plt.show()