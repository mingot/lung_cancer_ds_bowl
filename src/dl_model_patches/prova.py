import re
import json
import ast

idx = 0
with open('/Users/mingot/Projectes/kaggle/ds_bowl_lung/personal/debug_lung_segmentation/output_pre5.txt') as file:
    print 'patientid,fp,fn,tp'
    for row in file:
        idx += 1
        # if idx>10:break
        xx = re.match('.* Patient: (.*), stats: (.*)', row)
        if xx is not None:
            j = ast.literal_eval(xx.group(2))
            try:
                print "%s, %s, %s, %s" % (xx.group(1), j['fp'], j['fn'], j['tp'])
            except:
                print "%s,,," % xx.group(1)

