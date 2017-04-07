VilaDelPingui Data Science Bowl
Source Code Instructions

(1) Create a folder (Base Directory, BD) that will contain the models and temporary files.

(2) Store the 5 models in a subfolder BD/models/. The 5 models include: 2 DL for path location, 1 DL for malignancy detection
    2 GLM for feature aggregations.

(3) Configure the path variables in the following files:
     - master_execution.py:
            - INPUT_PATH: the path to the dicom files to be tested
            - BD: the directory created at (1) where the model and temporary files will be stored.
     - config.R:
            - BD: the directory created at (1) where the model and temporary files will be stored.

(4) Execute the script: 
> python master_execution.py

(5) The 2 model submission files will be stored in BD/output_csv/submission_file<01,02>.csv
