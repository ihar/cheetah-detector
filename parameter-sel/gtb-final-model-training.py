# From a grid search in gradient-tree-boosting.py we know an optimal parameter set.
# Apply it to all data in order to get final model.
# I personally needed the script to make a 32-bit model file

# Best score:
# 0.915222084348
# Best parameter set:
# {'loss': 'ls', 'verbose': 0, 'subsample': 1.0, 'learning_rate': 0.05, 'min_samples_leaf': 17, 'n_estimators': 3000, 'min_sa
# ples_split': 2, 'init': None, 'random_state': None, 'max_features': 0.1, 'alpha': 0.9, 'max_depth': 6}

from time import gmtime, strftime
import csv

import cPickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

import numpy as np

BOW_FEATURES = '../feature-gen/bow-500-500-50.csv'
LBP_FEATURES = '../feature-gen/lbp-1-8.csv'

FINAL_MODEL_FILE = './gtb-32bit.pkl'


def combine_features(bow_file, lbp_file):
    with open(bow_file, 'rb') as csvfile:
        bow_data = list(tuple(rec) for rec in csv.reader(csvfile, delimiter=','))
    bow_labels = np.array([int(seq[0]) for seq in bow_data])
    bow_filenames = [seq[1] for seq in bow_data]

    with open(lbp_file, 'rb') as csvfile:
        lbp_data = list(tuple(rec) for rec in csv.reader(csvfile, delimiter=','))
    # lbp_labels = np.array([int(seq[0]) for seq in lbp_data])
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
        descriptors.append(bow_row[2:] + lbp_row[2:])

    return labels, common_files, descriptors


if __name__ == '__main__':
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Read csv data and combine them..."
    labels, fnames, data = combine_features(BOW_FEATURES, LBP_FEATURES)

    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Convert numeric data to numpy arrays..."
    class_labels = np.array([int(item) for item in labels])
    data_mat = np.array(data).astype(np.float32)

    assert data_mat.shape[0] == class_labels.shape[0], 'Wrong dimensions of numpy arrays!'

    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Train selected model on the whole data set."

    # {'loss': 'ls', 'subsample': 1.0, 'learning_rate': 0.05, 'min_samples_leaf': 17, 'n_estimators': 3000, 'min_sa
    # ples_split': 2, 'init': None, 'random_state': None, 'max_features': 0.1, 'alpha': 0.9, 'max_depth': 6}
    final_gtb = GradientBoostingRegressor(learning_rate=0.05,
                                          n_estimators=3000,
                                          subsample=1.0,
                                          min_samples_split=2,
                                          min_samples_leaf=17,
                                          max_features=0.1,
                                          alpha=0.9,
                                          max_depth=6,
                                          loss='ls').fit(data_mat, class_labels)

    print roc_auc_score(class_labels, final_gtb.predict(data_mat))

    with open(FINAL_MODEL_FILE, 'wb') as fid:
        cPickle.dump(final_gtb, fid)

    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."

