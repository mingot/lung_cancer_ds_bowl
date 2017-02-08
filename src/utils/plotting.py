import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def multiplot(imgs):
    """Plot multiple imags in a grid."""
    nimg = len(imgs)
    num_rows = int(math.sqrt(nimg)) + 1
    f, plots = plt.subplots(num_rows, num_rows, sharex='all', sharey='all', figsize=(num_rows, num_rows))
    for i in range(nimg):
        plots[i // num_rows, i % num_rows].axis('off')
        plots[i // num_rows, i % num_rows].imshow(imgs[i])
        # plots[i // 11, i % 11].imshow(patient_slices[i], cmap=plt.cm.bone)


def plot_bb(img, regions):
    """Draw the img and the bounding box defined by a scikit image region (measure module)."""
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()

def plot_mask(img, mask):
    thr = np.where(mask < np.mean(mask),0.,1.0)  # threshold detected regions
    label_image = measure.label(thr)  # label them
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    plot_bb(img, regions)
    
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def cube_show_slider(cube, axis=2, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    from matplotlib.widgets import Slider, Button, RadioButtons

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    cube = cube.transpose(1, 2, 0)
    cube = cube[:-1:]

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in xrange(3)]
    im = cube[s].squeeze()

    # display image
    l = ax.imshow(im, **kwargs)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in xrange(3)]
        im = cube[s].squeeze()
        l.set_data(im, **kwargs)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()