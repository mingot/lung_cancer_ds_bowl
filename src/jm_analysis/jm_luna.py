from utils.reading import load_scan, get_pixels_hu, scan2imgs
from utils.segmentation import luna_segmentation, luna_apply_mask
from luna.LUNA03_train_unet import get_unet
from utils.features import extract_features_from_map
from skimage.transform import resize
from skimage import measure
import matplotlib.pyplot as plt
from glob import glob
from time import time
import SimpleITK as sitk

wp = os.environ['LUNG_PATH']
OUTPUT_FOLDER = wp + 'data/jm_luna_tmp/'
working_path = wp + 'data/jm_luna_tmp/'
MODEL_WEIGHTS = wp + 'models/unet_official.hdf5'

idx = 20
inp1 = np.load(working_path + 'images_%d.npy' % idx)
inp2 = np.load(working_path + 'lungmask_%d.npy' % idx)
inp3 = np.load(working_path + 'masks_%d.npy' % idx)
plt.imshow(inp1[0,:,:])
plt.imshow(inp2[0,:,:])
plt.imshow(inp3[0,:,:])



imgs_train = np.load(working_path + "trainImages.npy").astype(np.float32)
imgs_test = np.load(working_path + "testImages.npy").astype(np.float32)
imgs_test = np.concatenate((imgs_train, imgs_test))
num_test = len(imgs_test)
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
t_start = time()
for i in range(num_test):
    print 'Predicting img %d' % i
    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
np.save(OUTPUT_FOLDER + 'masksTestPredicted.npy', imgs_mask_test)
print 'Total time: %s' % (time()-t_start)






# patients = []
# for i in range(3):
#     patients += glob(wp + 'data/luna/subset%d/*.mhd' % i)

# # Load annotation nodes
# df_node = pd.read_csv(wp + "data/luna/annotations.csv")
# df_node = df_node.dropna()

# def matrix2int16(matrix):
#     ''' 
#     matrix must be a numpy array NXN
#     Returns uint16 version
#     '''
#     m_min= np.min(matrix)
#     m_max= np.max(matrix)
#     matrix = matrix-m_min
#     return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))


# def luna_slices(img_file):
#     mini_df = df_node[df_node["seriesuid"]==img_file.split('/')[-1].split('.mhd')[0]] #get all nodules associate with file
#     if len(mini_df)>0:    # some files may not have a nodule--skipping those 
#         biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
#         node_x = mini_df["coordX"].values[biggest_node]
#         node_y = mini_df["coordY"].values[biggest_node]
#         node_z = mini_df["coordZ"].values[biggest_node]
#         diam = mini_df["diameter_mm"].values[biggest_node]
#         # extracting image
#         itk_img = sitk.ReadImage(img_file)
#         img_array = sitk.GetArrayFromImage(itk_img) #indexes are z,y,x
#         num_z,height,width = img_array.shape        #heightXwidth constitute the transverse plane
#         imgs = np.ndarray([3,height,width],dtype=np.uint16)
#         masks = np.ndarray([3,height,width],dtype=np.uint8)
#         center = np.array([node_x,node_y,node_z])  #nodule center
#         origin = np.array(itk_img.GetOrigin())  #x,y,z  Origin in world coordinates (mm)
#         spacing = np.array(itk_img.GetSpacing())# spacing of voxels in world coor. (mm)
#         v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space

#         i = 0
#         for i_z in range(int(v_center[2])-1, int(v_center[2])+2):
#             imgs[i] = matrix2int16(img_array[i_z])
#             i+=1
#         return imgs
    
# model = get_unet()
# model.load_weights(MODEL_WEIGHTS)
# features_array = []
# for i, patient in enumerate(patients[0:3]):
    
#     # Load patient
#     t_start = time()
#     patient_slices = luna_slices(patient)
#     if not patient_slices:  # skip patients with no selected nodules
#         continue
#     nslices = len(patient_slices)
#     imgs_test = np.ndarray([len(slices_range),1,512,512],dtype=np.float32)
    
#     print "Creating mask for patient %s" % patient
#     t_mask = time()
#     idx = 0
#     for j in range(nslices):
#         img = patient_slices[j]
#         mask = luna_segmentation(img)
#         img3 = luna_apply_mask(img, mask)
#         imgs_test[idx,0] = img3
#         idx +=1
#     print 'Time for mask: %s' % str(time()-t_mask)
    
#     print "Applying DL for patient %s" % patient
#     t_dl = time()
#     num_test = len(imgs_test)
#     imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
#     for i in range(num_test):
#         print 'Predicting %d' %i
#         imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
#     print 'Time for DL: %s' % str(time()-t_dl)
    
#     #Store
#     np.save(TMP_FOLDER + 'dl_pred_' + patient + '.npy', imgs_mask_test)
    
#     # Extract high level features
#     print 'Extracting features for patient %s' % patient
#     t_feat = time()
#     features = extract_features_from_map(imgs_mask_test)
#     features_array.append(features)
#     print 'Time for mask: %s' % str(time()-t_feat)
    
#     print 'total time: %s' % str(time()-t_start)

