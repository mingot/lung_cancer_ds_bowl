from src.utils import reading
from src.utils import plotting
from glob import glob
import matplotlib.pyplot as plt


wp = os.environ['LUNG_PATH']

### LUNA

# load raw slices
luna_patients = glob(wp + 'data/luna/subset1/*.mhd')  # patients from subset1
patient_slices = reading.load_slices_from_mhd(luna_patients[0])
patient_slices.shape
plt.imshow(patient_slices[50,:,:])
plotting.plot_multiple_imgs(patient_slices[0:50,:,:])

# load selected slices (converted to int32)
idx = 0
im = np.load(wp + 'data/jm_luna_tmp/images_%d.npy' % idx)
im.shape
plt.imshow(im[1,:,:])

# load mask
mask = np.load(wp + 'data/jm_luna_tmp/masks_%d.npy' % idx)
mask.shape
plt.imshow(mask[0,:,:])

# load lung mask
lmask = np.load(wp + 'data/jm_luna_tmp/lungmask_%d.npy' % idx)
lmask.shape
plt.imshow(lmask[0,:,:])


# Load DSB