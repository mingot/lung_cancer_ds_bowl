


import pandas as pd
import matplotlib.pyplot as plt

df_node = pd.read_csv('/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/final_model/dl_unet_v01.csv')


DATA_PATH = "/Users/mingot/Projectes/kaggle/ds_bowl_lung/data/preprocessed5_sample/"
mylist = os.listdir(DATA_PATH)
file_list = [g for g in mylist if g.startswith('luna_')]
file_list

patient = np.load(DATA_PATH + file_list[0])['arr_0']
patient.shape

for patient in patient_list:  # to extract form .csv
    plt.imshow(patient[0,50])
    plt.show()
