from src.luna.LUNA03_train_unet import get_unet
from src.utils import reading
from src.utils import segmentation
from src.utils import plotting
from src.utils import features
import matplotlib.pyplot as plt
from time import time
from glob import glob
import numpy as np


wp = os.environ['LUNG_PATH']
TMP_FOLDER = wp + 'data/jm_tmp/'
INPUT_FOLDER = wp + 'data/sample_images/'
INPUT_FOLDER_BIG = wp + 'data/stage1/stage1/'
MODEL_WEIGHTS = wp + 'models/unet_official.hdf5'

# Load sample patients
patients = os.listdir(INPUT_FOLDER)  # sample

# Load all patients
# patients = os.listdir(INPUT_FOLDER_BIG)  # all patients

# Load previous patients
# prev_patients = os.listdir(TMP_FOLDER)
# prev_patients = [x.split('.')[0].split('_')[-1] for x in prev_patients]
# patients = [x for x in patients2 if x not in prev_patients]

patients.sort()



######################
# Read and apply DL model to images to extract nodule heatmap
#####################

model = get_unet()
model.load_weights(MODEL_WEIGHTS)
features_array = []
for i, patient in enumerate(patients):
    
    # i=0
    # patient = patients[0]
    
    # Load patient
    t_start = time()
    patient_slices = reading.scan2imgs(reading.load_scan(INPUT_FOLDER_BIG + patient))
    nslices = len(patient_slices)
    
    # select slices for which to compute the heatmap (middle 30%)
    slices_range = range(int(nslices*0.3), int(nslices*0.6))
    slices_range = [x for x in slices_range if x%4==0]  # reduce slices
    imgs_test = np.ndarray([len(slices_range),1,512,512],dtype=np.float32)
    #img = patient_slices[60]
    
    print "Creating mask for patient %s" % patient
    t_mask = time()
    idx = 0
    for j in slices_range:
        img = patient_slices[j]
        mask = segmentation.luna_segmentation(img)
        img3 = segmentation.luna_apply_mask(img, mask)
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


# checks 1
imgs_mask_test.shape
plt.imshow(imgs_mask_test[0,0,:,:])
plt.imshow(patient_slices[40])

plotting.multiplot(patient_slices[slices_range])
plotting.multiplot(imgs_mask_test[slices_range,0,:,:])


# checks 2: Compare raw images
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



######################
# Create training data set
######################

labels = pd.read_csv(wp + 'data/stage1_labels.csv') 
nodfiles = glob(wp + 'data/jm_dsb_tmp/*.npy')
nodfiles = [x for x in nodfiles if x.split("_")[-1].split('.npy')[0] in list(labels['id'])]  # remove users from testset 

num_features = 9
feature_array = np.zeros((len(nodfiles), num_features))
truth_metric = np.zeros((len(nodfiles)))
for i,nodfile in enumerate(nodfiles):
    print 'Extracting features for patient %d' % i
    pat_id = nodfile.split("_")[-1].split('.npy')[0]
    truth_metric[i] = int(labels[labels['id']==pat_id]['cancer'])
    feature_array[i] = features.extract_features_from_map(slices_img_map=np.load(nodfile))
    
np.save("dataY.npy", truth_metric)
np.save("dataX.npy", feature_array)

plt.imshow(np.load(nodfile)[3,0,:,:])


######################
# Train and evaluate XGBoost
######################


from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from skimage import measure
import scipy as sp

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

X = feature_array  # np.load("dataX.npy")
Y = truth_metric  # np.load("dataY.npy")
kf = KFold(Y, n_folds=3)
y_pred = Y * 0
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
    clf = xgb.XGBClassifier(objective="binary:logistic")
    clf.fit(X_train, y_train)
    y_pred[test] = 0.0 # clf.predict(X_test)
print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
print("logloss",logloss(Y, y_pred))


np.mean(Y)  # 23%

# Random Forest
#              precision    recall  f1-score   support
#   No Cancer       0.81      0.98      0.89       463
#      Cancer       0.17      0.02      0.03       107
# avg / total       0.69      0.80      0.73       570
# ('logloss', 0.52600332670816652)
# XGBoost
#              precision    recall  f1-score   support
#   No Cancer       0.83      0.86      0.84       463
#      Cancer       0.27      0.21      0.24       107
# avg / total       0.72      0.74      0.73       570
# ('logloss', 0.5700685138621493)
# Predicting all positive
#              precision    recall  f1-score   support
#   No Cancer       0.00      0.00      0.00       463
#      Cancer       0.19      1.00      0.32       107
# avg / total       0.04      0.19      0.06       570
# ('logloss', 28.055831025357818)
# Predicting all negative
#              precision    recall  f1-score   support
#   No Cancer       0.81      1.00      0.90       463
#      Cancer       0.00      0.00      0.00       107
# avg / total       0.66      0.81      0.73       570
# ('logloss', 6.4835948671148085)





