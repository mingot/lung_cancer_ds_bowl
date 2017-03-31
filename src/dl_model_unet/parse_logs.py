import string
import re

f = open('out4.txt')
lines = f.readlines()
f.close()

lines2 = [l.translate(string.maketrans('\b',' ')) for l in lines]
lines3 = [l.strip() for l in lines2 if 'loss' in l]

log = []
for line in lines3:
	s=re.sub(r'(\d\d-\d\d \d\d:\d\d:\d\d) .*',r' - time: \1',line)
	s=re.sub(r'(\d\d?\d?/300) \[[^ ]* -', r'batch: \1 -' ,s)
	ss=s.split(' - ')
	d=[v.split(': ') for v in ss]
	d = [x for x in d if len(x)==2]
	log.append(dict(d))

import pandas as pd

X=pd.DataFrame.from_records(log)

X.to_csv('log.csv')

