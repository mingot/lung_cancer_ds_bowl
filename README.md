# Lung Cancer Data Science Bowl 2017

## Introduction
Repository for the Vila del Pingui team for the Data Science Bowl 2017 (Feb2017 to Apr2017). The competetition ($1M in prizes) was about predicting early stage lung cancer from CT Scan images. The training set was 1397 + 200 patients and the test 500 patients. The result is an ensemble of 3 convolutional neural networks (resnet) for feature generation and xgboost for final ensemble.  

The team ended in 34th position of 2000 teams (top 2%) with the best model scoring in the 17th position.

## Index
Access to latest results of each team and to documentation

  1. Preprocessing and datasets (README TBD)
    1. [Utils (git)](https://github.com/mingot/lung_cancer_ds_bowl/tree/master/src/utils) 
    2. [Bad segmentation spreadsheet (gdocs)](https://docs.google.com/spreadsheets/d/15wi07edzdVLqpnviPI4qhO5gOA_Ve6MQ7RCE92YGERE/edit#gid=0)
    3. Preprocessed v3 (AWS): /mnt/hd2/preprocessed3 
  2. DL ([README](https://github.com/mingot/lung_cancer_ds_bowl/blob/master/src/jc_dl/README.md))
    1. Slices: TBD
    2. Segmentation: TBD
  3. Final model (README TBD)
    1. New features: TBD
    2. XGBoost: TBD
    3. Final learner - submission: TBD
  4. Literature: 
    1. Preprocessing ([google drive](https://drive.google.com/drive/folders/0BwtD1eiXdLQRVXo4aXFYVDVZNHM))
    2. DL ([google drive](https://drive.google.com/drive/folders/0BwtD1eiXdLQRMlhvTzJsZjNkeWs))
    3. Features ([google drive](https://drive.google.com/drive/folders/0BwtD1eiXdLQRS1d3eVlwSVpCblU))

## References quick start
Basic references to understand the problem and the data:

 1. [Video] (https://www.youtube.com/watch?v=-XUKq3B4sdw) how to detect a lung cancer from a physician perspective (15 min).
 2. Notebooks (Kaggle Kernels) Understand the data set and dealing with DICOM files. 
   1. [Preprocessing tutorial](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial): understanding DICOM files, pixel values, standarization, ...
   2.  [Exploratory data analysis](https://www.kaggle.com/anokas/data-science-bowl-2017/exploratory-data-analysis): basic exploration of the given data set
 3. [Kaggle tutorial] (https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial) with code for training a CNN using the U-net network for medical image segmentation. Based on the external LUNA data set (annotated).
 4. [TensorFlow ppt] (https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/edit#slide=id.p) for quickstart (focused on convnets) and code included. After it, you can take the [official TF tutorial](https://www.tensorflow.org/tutorials/deep_cnn/) as the sample code.

## Quickstart
 [TBD]
 1 - Download the repo:
 ```
 $ git clone https://github.com/mingot/lung_cancer_ds_bowl.git
 ```
 2 - Create [virtual enviroment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) (see `virtualenvwrapper`) and install python requirements
```
$ mkvirtualenv lung
$(lung) pip install -r requirements.txt
```

## Jupyter
 - Estan ja instalats els paquets de `requirements.txt` amb el kernel de python2.
 - Cada usuari pot fer git pull/commit/push desde un ssh o amb `!git commit` .. desde la consola de jupyter. No demana contrasenya, el usuari queda identificat amb el email
 - Cada usuari té el seu directori `~/lung_cancer_ds_bowl` privat per ell excepte la carpeta `~/lung_cancer_ds_bowl/data` que es compartida per tots.
 - Tots els usuaris tenen permís de sudo així que si cal instalar paquets poden fer servir `!sudo pip install` paquet desde jupyter i així seràn accesibles per tots.

## Available datasets
See docs/ 

## Preprocessing
The preprocessed images are stored at `/mnt/hd2/preprocessed/`. To open the compressed files from python use the following instruction:
`np.load(file)['arr_0']`. There is one file per patient. Eah file is a numpy array of 4 dimensions: `[type,slice,height,width]`. The dimension `type` contains the preprocessed image at index 0, the lung segmentation at index 1, and when available (luna dataset) the nodules segmentation at index 2. All the images have dimensions `height` and `weight` dimensions of 512x512.

## General guidelines
 - The analysis files should start with the author initials.
 - Avoid storing files >50Mb in Git. In particular, images from data folder should be outside the git repository.

## File structure


```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

## Troubleshoot

### "Could not find a version that satisfies the requirement SimpleITK==0.10.0"

The solution is to manually download the egg from the [official website](https://itk.org/SimpleITKDoxygen/html/PyDownloadPage.html) and install it with `easy_install`.

### "Fatal Python error: PyThreadState_Get: no current thread"
```
>>> import SimpleITK as sitk
"Fatal Python error: PyThreadState_Get: no current thread"
```
The solution is to relink the `_SimpleITK.so`:
```
$ otool -L ~/virtualenvs/lung/lib/python2.7/site-packages/SimpleITK/_SimpleITK.so 
/Users/mingot/virtualenvs/lung/lib/python2.7/site-packages/SimpleITK/_SimpleITK.so:
	/System/Library/Frameworks/Python.framework/Versions/2.7/Python (compatibility version 2.7.0, current version 2.7.1)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 635.19.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 159.1.0)
	/usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 52.0.0)
$ sudo install_name_tool -change /System/Library/Frameworks/Python.framework/Versions/2.7/Python ~/virtualenvs/lung/.Python ~/virtualenvs/lung/lib/python2.7/site-packages/SimpleITK/_SimpleITK.so
```
