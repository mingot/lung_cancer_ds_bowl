# xx = np.load(wp + 'data/preproc_luna/luna_100684836163890911914061745866.npz')['arr_0']
# xx = np.load(wp + 'data/preproc_dsb/dsb_0a099f2549429d29b32f349e95fb2244.npz')['arr_0']
# plt.imshow(img[80])
# img[80]

# img.shape
# a = np.zeros((img.shape[0],512, 512))
# a.fill(-0.25)
# a.shape
# img.shape
# x = (512 - img.shape[1])/2
# y = (512 - img.shape[2])/2
# a[:, x:x+img.shape[1], y:y+img.shape[2] ] = img
# plt.imshow(a[200])


## CHECKS for differences luna <> dsb
# import SimpleITK as sitk
# luna_patients = glob(wp + 'data/luna/subset1/*.mhd')  # patients from subset1
# img_file = luna_patients[0]
# itk_img = sitk.ReadImage(img_file)
# img_array = sitk.GetArrayFromImage(itk_img) #indexes are z,y,x
# itk_img.GetSpacing()
# patients = reading.load_scan(os.path.join(INPUT_FOLDER, patients[0]))
# patients
# spacing = map(float, ([patients[0].SliceThickness] + patients[0].PixelSpacing))
# spacing
