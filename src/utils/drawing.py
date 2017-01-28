import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_bb(img, region):
    """Draw the img and the bounding box defined by a scikit image region (measure module)."""
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    plt.show()
