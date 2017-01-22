# Lung Cancer Data Science Bowl 2017


## Quickstart


## General guidelines
 - By default, every file should start with the main author initials.
 - Avoid storing files >50Mb in Git. In particular, images from data folder should be outside the git repository.


Data
Luna: download images from Luna Challenge


## File structure

```
├── Makefile           <- Makefile with commands like `make data` or `make train`
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
