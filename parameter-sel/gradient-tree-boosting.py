from time import gmtime, strftime
import csv
import os.path

import cPickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

import numpy as np

BOW_FEATURES = '../feature-gen/bow-500-500-50.csv'
LBP_FEATURES = '../feature-gen/lbp-1-8.csv'

COARSE_MODEL_FILE = './coarse-gtb.model'
TUNED_MODEL_FILE = './tuned-gtb.model'
FINAL_MODEL_FILE = './final-gtb.model'

# Load distinct feature sets from two separate files,
# make a combined feature for each image file.
# Return
# 1) a list of class labels (0 or 1)
# 2) a list of filenames
# 3) an array of combined features
def combine_features(bow_file, lbp_file):
    with open(bow_file, 'rb') as csvfile:
        bow_data = list(tuple(rec) for rec in csv.reader(csvfile, delimiter=','))
    bow_labels = np.array([int(seq[0]) for seq in bow_data])
    bow_filenames = [seq[1] for seq in bow_data]

    with open(lbp_file, 'rb') as csvfile:
        lbp_data = list(tuple(rec) for rec in csv.reader(csvfile, delimiter=','))
    #lbp_labels = np.array([int(seq[0]) for seq in lbp_data])
    lbp_filenames = [seq[1] for seq in lbp_data]

    # Find a correspondence between rows of the two above files
    common_files = list(set(bow_filenames) & set(lbp_filenames))
    labels = []
    descriptors = []
    for fname in common_files:
        bow_row = bow_data[bow_filenames.index(fname)]
        lbp_row = lbp_data[lbp_filenames.index(fname)]
        assert bow_row[0] == lbp_row[0], 'Class labels are not the same for the same files!'
        labels.append(bow_row[0])
        descriptors.append(bow_row[2:]+lbp_row[2:])

    return labels, common_files, descriptors

def read_gtb_model_from_file(fname):
    with open(fname, 'rb') as fid:
        gtb_loaded = cPickle.load(fid)
    return gtb_loaded

if __name__ == '__main__':
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Read csv data and combine them..."
    labels, fnames, data = combine_features(BOW_FEATURES, LBP_FEATURES)

    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Convert numeric data to numpy arrays..."
    class_labels = np.array([int(item) for item in labels])
    data_mat = np.array(data).astype(np.float32)

    assert data_mat.shape[0] == class_labels.shape[0], 'Wrong dimensions of numpy arrays!'

    if (not os.path.exists(COARSE_MODEL_FILE)):
        # Find best parameter set for gradient boosted trees
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Find a semi-optimal parameter set for GBT..."
        param_grid = {'learning_rate': [0.2, 0.1, 0.05],
                      'max_depth': [4, 6],
                      'min_samples_leaf': [3, 9, 17],
                      'max_features': [1.0, 0.3, 0.1]}
        # param_grid = {'learning_rate': [0.1],
        #               'max_depth': [4],
        #               'min_samples_leaf': [9],
        #               'max_features': [1.0]}
        sss = StratifiedShuffleSplit(class_labels, 5, test_size=0.25, random_state=530)
        gscv = GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=1000), param_grid=param_grid, scoring='roc_auc', n_jobs=4, cv=sss, verbose=2)
        gscv.fit(data_mat, class_labels)

        print "Best parameters set found on cv set:"
        print gscv.best_estimator_
        print "Best score:"
        print gscv.best_score_
        print "Best parameter set:"
        print gscv.best_params_
        print ""
        print("Scores for all parameter combinations:")
        for params, mean_score, scores in gscv.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

        with open(COARSE_MODEL_FILE, 'wb') as fid:
            cPickle.dump(gscv.best_estimator_, fid)

    if (not os.path.exists(TUNED_MODEL_FILE)):
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Tune a model further, increase number of trees and check wider range for learning rate..."
        # File with a coarse model exists, load it and perform further tuning.
        # Set n_estimators higher (3000-4000) and tune learning_rate again
        coarse_gbt = read_gtb_model_from_file(COARSE_MODEL_FILE)
        print "Current parameters of the estimator:"
        print coarse_gbt.get_params()
        param_grid = {'learning_rate': [0.5, 0.2, 0.1, 0.05, 0.025, 0.01],
                      'n_estimators': [2000, 3000]}
        sss = StratifiedShuffleSplit(class_labels, 5, test_size=0.25, random_state=530)
        gscv = GridSearchCV(estimator=coarse_gbt, param_grid=param_grid, scoring='roc_auc', n_jobs=4, cv=sss, verbose=2)
        gscv.fit(data_mat, class_labels)
        print "Best parameters set found on cv set:"
        print gscv.best_estimator_
        print "Best score:"
        print gscv.best_score_
        print "Best parameter set:"
        print gscv.best_params_
        print ""
        print("Scores for all parameter combinations:")
        for params, mean_score, scores in gscv.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

        with open(TUNED_MODEL_FILE, 'wb') as fid:
            cPickle.dump(gscv.best_estimator_, fid)

    # And now train the best model on the whole data set
    if (not os.path.exists((FINAL_MODEL_FILE))):
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Train selected model on the whole data set."
        tuned_gtb = read_gtb_model_from_file(TUNED_MODEL_FILE)
        print tuned_gtb.get_params()
        tuned_gtb.fit(data_mat, class_labels)
        with open(FINAL_MODEL_FILE, 'wb') as fid:
            cPickle.dump(tuned_gtb, fid)

    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."