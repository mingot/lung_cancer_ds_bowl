import re
import json
import ast

# idx = 0
# with open('/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/debug_lung_segmentation/output_pre5.txt') as file:
#     print 'patientid,fp,fn,tp'
#     for row in file:
#         idx += 1
#         # if idx>10:break
#         xx = re.match('.* Patient: (.*), stats: (.*)', row)
#         if xx is not None:
#             j = ast.literal_eval(xx.group(2))
#             try:
#                 print "%s, %s, %s, %s" % (xx.group(1), j['fp'], j['fn'], j['tp'])
#             except:
#                 print "%s,,," % xx.group(1)


import os

INPUT_PATH = '/mnt/hd2/preprocessed5'  # INPUT_PATH = wp + 'data/preprocessed5_sample'
VALIDATION_PATH = '/mnt/hd2/preprocessed5_validation_luna'


f = open('/home/aitor/lung_cancer_ds_bowl/output/nodules_patches_dl1_v11.csv')

file_list = [os.path.join(VALIDATION_PATH, fp) for fp in os.listdir(VALIDATION_PATH)]
file_list += [os.path.join(INPUT_PATH, fp) for fp in os.listdir(INPUT_PATH)] # if fp.startswith('dsb_')]



nslice_old, fidx = 0, 0
patientid = file_list[fidx].split('/')[-1]

for idx,row in enumerate(f):
    if idx==0:
        print row
        continue
    if idx>10:
        break
    row = row.split(',')
    nslice = row[1]
    if nslice>=nslice_old:
        print "%s,%s" % (patientid,','.join(row[1:]))
    else:
        fidx += 1
        patientid = file_list[fidx].split('/')[-1]
        nslice_old = nslice
        print "%s,%s" % (patientid,','.join(row[1:]))

