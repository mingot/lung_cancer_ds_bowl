# Datasets

## Summary

| Name | Type  | # patients | slices per patient | Voxel Size (mm) | File Size (GB) |
| :--- |:---: | :---: | :---: | :---: | :---: | :---: |
| [DSB](https://www.kaggle.com/c/data-science-bowl-2017) | Lung Classification | 1397 | ~ 180 | ?x?x? - ?x?x? | ~ 140 | 
| [LIDC/IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | Annotated Nodules | 1018 | < ~ 250 | ?x?x? - ?x?x? | ~ 125 |
| [LUNA16](https://luna16.grand-challenge.org) | Nodule Location | 888 | ~ 250 | ?x?x0.8 - ?x?x2.5 | ~ 120 |
| [SPIE-AAPM](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge) | Nodule Classification | 70 | ? | ?x?x? - ?x?x2.5 | ~ 12 |
| [ANODE09](https://anode09.grand-challenge.org) | Nodule Location | 50 | ? | ?x?x? - ?x?x? | ~ 5* |
| [ELCAP](http://www.via.cornell.edu/lungdb.html) | Nodule Location | 50 | ~ 260 | ?x?x1.25 - ?x?x1.25 | ~ 4* |
| [EmphysemaDB](http://image.diku.dk/emphysema_database) | Emphysema Classification | 39 | ~ 3 | 0.78x0.78x1.25 | ~ 0.04* |
\*Compressed (probably x1.5-2 ratio)

They all have different resultions/sizes, normalization will be needed for inter and intra dataset analysis!

## Extra Information

### DSB

Our official competition data.

### LIDC/IDRI

Biggest public CT's external db. Annotations are NOT binary (malign/benign), there are some subtleties.

### LUNA16

Subset of the LIDC/IDRI dataset with a thickness lesser or equal to 2.5mm and excluding *non-relevant* nodules annotations.

### SPIE-AAPM

Small dataset with benign/malign nodule classification dataset that can also be used for detection.

### ANODE09

Small dataset.

### ELCAP Public Lung Image Database

Small dataset.

### Emphysema Database

Small dataset. *Emphysema* is a risk factor for lung cancer.
