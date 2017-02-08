
import os
from utils import plotting
from utils import reading
from utils import preprocessing
import random
from glob import glob



## Loading
wp = os.environ['LUNG_PATH']
OUTPUT_FOLDER  = wp + 'data/preproc_luna/*.npz'  # Test DSB
OUTPUT_FOLDER  = wp + 'data/preproc_dsb/*.npz'  # Test luna


proc_file = glob(OUTPUT_FOLDER)  # patients from subset1
pfile = random.choice(proc_file)
p = np.load(pfile)['arr_0']
p.shape


# 3d sliding plot
plotting.cube_show_slider(p[0])  # image
plotting.cube_show_slider(p[1])  # lung mask
plotting.cube_show_slider(p[2])  # nodule mask (if exists)


# histogram
plt.hist(p[0,5].flatten(), bins=80)
plt.show()

# study nodules mask
for i in range(p[2].shape[0]):
    if np.max(p[2,i])!=0:
        print i

plt.imshow(p[2,85])
plt.show()

plotting.plot_mask(p[0,85], p[2,85])
proc_file
