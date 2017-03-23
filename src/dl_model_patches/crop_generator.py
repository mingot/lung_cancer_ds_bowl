import os


filenames


X_train, y_train = [], []
for idx,filename in enumerate(filenames_train):
    patientid = filename.split('/')[-1]
    logging.info("Progress %d/%d" % (idx,len(filenames_train)))
    X_single, y_single = load_patient_with_candidates(filename, nodules_df[nodules_df['patientid']==patientid], thickness=1)
    X_train.extend(X_single)
    y_train.extend(y_single)
