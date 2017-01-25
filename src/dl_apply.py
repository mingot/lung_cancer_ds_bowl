from src.utils.images import load_scan, get_pixels_hu
from src.utils.segmentation import luna_segmentation, luna_apply_mask
from skimage.transform import resize
from skimage import measure
import matplotlib.pyplot as plt
from glob import glob


INPUT_FOLDER = os.environ['LUNG_PATH'] + 'data/sample_images/'
MODEL_WEIGHTS = os.environ['LUNG_PATH'] + 'models/unet_official.hdf5'

# Load patients
patients = os.listdir(INPUT_FOLDER)
patients.sort()

def scan2imgs(scans):
    imgs = np.stack([s.pixel_array for s in scans])
    imgs = imgs.astype(np.int16)
    imgs = np.array(imgs, dtype=np.int16)
    return imgs

# Process images
imgs_test = np.ndarray([len(imgs),1,512,512],dtype=np.float32)
for i, patient in enumerate(patients):
    #patient_pixels = get_pixels_hu(load_scan(INPUT_FOLDER + patient))
    patient_pixels = scan2imgs(load_scan(INPUT_FOLDER + patient))
    img = patient_pixels[60]
    mask = luna_segmentation(img)
    img3 = apply_mask(img, mask)
    imgs_test[i,0] = img3
    

# load initial images to compare

# Compare raw
file_list = glob(os.environ['LUNG_PATH'] + "data/luna_output/images_*.npy")
img_old = np.load(file_list[0]).astype(np.float64)[0] 
scans = load_scan(INPUT_FOLDER + patients[0])
patient_pixels = scan2imgs(scans)
img_new = patient_pixels[60]
plt.imshow(img_old)
plt.imshow(img_new)


# Compare post proces
imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
img_old = imgs_train[4,0]
img_new = imgs_test[4,0]
np.std(img_old)
np.std(img_new)

plt.imshow(img_old)
plt.imshow(img_new)

# Predict cancer heatmap
model = get_unet()
model.load_weights(MODEL_WEIGHTS)
num_test = len(imgs_test)
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
for i in range(num_test):
    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]


plt.imshow(imgs_test[1,0])
plt.imshow(imgs_mask_test[1,0])




# Exploratory
first_patient = load_scan(INPUT_FOLDER + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[82], cmap=plt.cm.gray)
plt.show()
