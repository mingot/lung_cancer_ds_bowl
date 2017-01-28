from src.utils.images import load_scan, get_pixels_hu, scan2imgs
from src.utils.segmentation import luna_segmentation, luna_apply_mask
from src.luna.LUNA03_train_unet import get_unet
from src.utils.features import extract_features_from_map
from skimage.transform import resize
from skimage import measure
import matplotlib.pyplot as plt
from glob import glob
from time import time


TMP_FOLDER = os.environ['LUNG_PATH'] + 'data/jm_tmp/'
INPUT_FOLDER = os.environ['LUNG_PATH'] + 'data/sample_images/'
MODEL_WEIGHTS = os.environ['LUNG_PATH'] + 'models/unet_official.hdf5'

# Load patients
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# Process images
#imgs_test = np.ndarray([len(patients),1,512,512],dtype=np.float32)
model = get_unet()
model.load_weights(MODEL_WEIGHTS)
features_array = []
for i, patient in enumerate(patients):
    
    # Load patient
    t_start = time()
    patient_slices = scan2imgs(load_scan(INPUT_FOLDER + patient))
    nslices = len(patient_slices)
    slices_range = range(int(nslices*0.3), int(nslices*0.6))
    slices_range = [x for x in slices_range if x%4==0]  # reduce slices
    imgs_test = np.ndarray([len(slices_range),1,512,512],dtype=np.float32)
    #img = patient_slices[60]
    
    print "Creating mask for patient %s" % patient
    t_mask = time()
    idx = 0
    for j in slices_range:
        img = patient_slices[j]
        mask = luna_segmentation(img)
        img3 = luna_apply_mask(img, mask)
        imgs_test[idx,0] = img3
        idx +=1
    print 'Time for mask: %s' % str(time()-t_mask)
    
    print "Applying DL for patient %s" % patient
    t_dl = time()
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        print 'Predicting %d' %i
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    print 'Time for DL: %s' % str(time()-t_dl)
    
    #Store
    np.save(TMP_FOLDER + 'dl_pred_' + patient + '.npy', imgs_mask_test)
    
    # Extract high level features
    print 'Extracting features for patient %s' % patient
    t_feat = time()
    features = extract_features_from_map(imgs_mask_test)
    features_array.append(features)
    print 'Time for mask: %s' % str(time()-t_feat)
    
    print 'total time: %s' % str(time()-t_start)


imgs_mask_test.shape
plt.imshow(imgs_mask_test[40,0,:,:])
plt.imshow(patient_slices[4])

plot_multiple_imgs(patient_slices[slices_range])
plot_multiple_imgs(imgs_mask_test[slices_range,0,:,:])


# Create training data set
feature_array = np.zeros((len(nodfiles),numfeatures))
truth_metric = np.zeros((len(nodfiles)))
    
xx = extract_features_from_map(imgs_mask_test)    
    
for i,nodfile in enumerate(nodfiles):
    patID = nodfile.split("_")[2]
    truth_metric[i] = truthdata[int(patID)]
    feature_array[i] = getRegionMetricRow(nodfile)
    
np.save("dataY.npy", truth_metric)
np.save("dataX.npy", feature_array)


# try XGBoost
print ("XGBoost")
X = feature_array  # np.load("dataX.npy")
Y = truth_metric  # np.load("dataY.npy")
kf = KFold(Y, n_folds=3)
y_pred = Y * 0
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
    clf = xgb.XGBClassifier(objective="binary:logistic")
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict(X_test)
print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
print("logloss",logloss(Y, y_pred))




# Compare raw images
file_list = glob(os.environ['LUNG_PATH'] + "data/luna_output/images_*.npy")
img_old = np.load(file_list[0]).astype(np.float64)[0] 
scans = load_scan(INPUT_FOLDER + patients[0])
patient_pixels = scan2imgs(scans)
img_new = patient_pixels[60]
plt.imshow(img_old)
plt.imshow(img_new)


# Compare post process images
imgs_train = np.load(os.environ['LUNG_PATH'] + "data/luna_output/trainImages.npy").astype(np.float32)
img_old = imgs_train[4,0]
img_new = imgs_test[4,0]
plt.imshow(img_old)
plt.imshow(img_new)


# Predict cancer heatmap
model = get_unet()
model.load_weights(MODEL_WEIGHTS)
num_test = len(imgs_test)
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
for i in range(num_test):
    print 'Predicting %d' %i
    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]


plt.imshow(imgs_test[1,0])
plt.imshow(imgs_mask_test[1,0])

slice_npy = imgs_mask_test[1,0]



