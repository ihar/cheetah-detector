#!/usr/bin/env python

# Errors' codes
#1   Error: can't find a codebook in the working directory
#2   Error: can't find a model file in the working directory
#3   Error: can't load the codebook   
#4   Error: can't load the model
#5   Error: unknown source of input data
#6   Error: too few arguments
#7   Error: too many arguments

import logging  # Tracking events that happen when the application runs
from time import gmtime, strftime  # Date-time for logging messages
import sys  # For an error message
import argparse  # Command line arguments
import pprint  # Pretty print some dictionaries (parameters of codebook and model)

# It is to resolve windows executable dependencies
from scipy import sparse
from scipy import linalg

import sklearn

# Reading file list from a directory or from a text file
from os import listdir
from os.path import isdir, exists, isfile, join

import cPickle

import numpy as np  # Dealing with feature vectors

import csv  # Write result of prediction to a comma separated file

# Image processing, SURF features
import cv2
# LBP features
from mahotas import features

CODEBOOK_FILE = './codebook.pkl'
MODEL_FILE = './gtb.pkl'

logger = logging.getLogger('cheetah-detector')

# Disable logging, because when GUI will be added, the logging won't be necessary
logger.disabled = True

# logger.setLevel(logging.DEBUG)
# # In the file we will log detail information
# log_fname = strftime("%Y-%m-%d", gmtime())
# fh = logging.FileHandler(log_fname)
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

# Prints a message to stderr and exits
def error(message, code):
    sys.stderr.write(message + "\n")
    sys.exit(code)

# Extracts full file names of images from a directoty
# Test path: d:\cheetah_trials\gbm_detected_bad
def extract_image_paths(dir_path):
    flist = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return flist

# Check if all necessary files exist and load them.
# We need precalculated codebook and a model file
def load_model_files(codebook_file, gtb_model_file):
    if not exists(codebook_file):
        logger.error('Can\'t find a codebook in the working directory')
        error("Error: can't find a codebook in the working directory", 1)
    if not exists(gtb_model_file):
        logger.error('Can\'t find a model file in the working directory')
        error("Error: can't find a model file in the working directory", 2)

    with open(codebook_file, 'rb') as fid:
        try:
            codebook = cPickle.load(fid)
            logger.debug('Load a codebook')
        except:
            logger.error('Can\'t load the codebook ' + codebook_file)
            error('Error: can\'t load the codebook ' + codebook_file, 3)

    with open(gtb_model_file, 'rb') as fid:
        try:
            model = cPickle.load(fid)
            logger.debug('Load a model file')
        except:
            logger.error('Can\'t load the model ' + gtb_model_file)
            error('Error: can\'t load the model ' + gtb_model_file, 4)

    return codebook, model

# Preprocess image
def preprocess_img(image_in):
    # Preprocessing step: histogram equalization and reduced image size proportionally to width 300 pixels
    img = cv2.equalizeHist(image_in)
    scale_factor = 300.0 / img.shape[:2][1]
    small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    return small_img

# Extract image features
# feature_name = {sift, surf, orb, brisk}
def extract_img_feature(img, feature_name):
    if feature_name == 'sift':
        detector = cv2.SIFT()  # Not tested
    elif feature_name == 'surf':
        detector = cv2.SURF(700)
    elif feature_name == 'orb':
        detector = cv2.ORB(10)  # Not tested
    elif feature_name == 'brisk':
        detector = cv2.BRISK(50)  # Not tested
    else:
        # Will not happen now because we define the feature right in the code
        sys.stdout.write("Unknown image feature: " + feature_name + "\n")
        logger.warning('Unknown image feature: ' + feature_name)
        return None
    kp, des = detector.detectAndCompute(img, None)
    return des

# Find BoW for an image
def get_bow(codebook, img_path, feature_name, cluster_count):
    img_features = extract_img_feature(img_path, feature_name)
    if (img_features is None) or (codebook.cluster_centers_.shape[1] != img_features.shape[1]):
        return None
    pred = codebook.predict(img_features)
    hist, bin_edges = np.histogram(pred, bins=cluster_count, range=(1, cluster_count), normed=True)
    return hist

# Extract histogram of lbp
def extract_lbp_hist(img, r, n):
    return features.lbp(img, r, n)

# Get feature vector for an image
def get_feature_vector(codebook, image_path):
    img = cv2.imread(image_path, 0)
    if (img is None) or (0 == img.size):
        logger.warning('Can\'t read the file ' + image_path)
        sys.stderr.write('Can\'t read the file ' + image_path + "\n")
        return None
    img = preprocess_img(img)
    bow = get_bow(codebook, img, 'surf', codebook.n_clusters)
    if (bow is None) or (len(bow) != codebook.n_clusters):
        logger.warning('Bad BoW descriptor for image ' + image_path)
        sys.stderr.write('Bad BoW descriptor for image ' + image_path + "\n")
        return None
    lbp = extract_lbp_hist(img, 1, 8)
    if lbp is None:
        logger.warning('Bad LBP descriptor for image ' + image_path)
        sys.stderr.write('Bad LBP descriptor for image ' + image_path + "\n")
        return None

    res = np.array(list(bow) + list(lbp)).astype(np.float32)
    return res

# Given a model file and a feature vector, make prediction
def make_prediction(model, feature):
    if (np.isnan(feature).any() or np.isinf(feature).any()):
        return None
    pred = model.predict(feature)
    # Dealing with values outside the range [0, 1]
    res = min(1, max(0, pred[0]))
    return res

if __name__ == '__main__':

    logger.info('Start the application')
    logger.debug('Number of arguments: ' + str(len(sys.argv)))
    logger.debug('Argument list: ' + str(sys.argv))
    # logger.debug('Codebook file name: ' + CODEBOOK_FILE)
    # logger.debug('Model file name: ' + GTB_MODEL_FILE)

    input_source = ''
    output_file = ''

    parser = argparse.ArgumentParser(description='Recognizes images with spotted cats')
    parser.add_argument("input", \
                        help='Either path to a directory with images or a text file containing paths to images')
    # parser.add_argument("output", \
    #                     help='Name of output file')

    # TODO: prints 'None' at the end of the help message, fix it
    # print parser.print_help()

    if len(sys.argv) < 2:
        error("Error: too few arguments", 6)
    if len(sys.argv) > 2:
        error("Error: too many arguments", 7)

    args = parser.parse_args()
    input_source = args.input
    # output_file = args.output

    logger.debug("Input source: " + input_source)
    # logger.debug("Output file: " + output_file)
    logger.debug('Codebook file name: ' + CODEBOOK_FILE)
    logger.debug('Model file name: ' + MODEL_FILE)

    if isdir(input_source):
        flist = extract_image_paths(input_source)
    elif isfile(input_source):
        with open(input_source) as f:
            flist = f.readlines()
        flist = [x.strip('\n') for x in flist]
    else:
        logger.error('Unknown source of input data')
        error('Error: unknown source of input data', 5)

    flist_len = len(flist)
    sys.stdout.write("Files to process: " + str(flist_len) + "\n")
    logger.debug('Number of files to process: ' + str(flist_len))

    # Read every image from the list, generate a feature vector
    # and make prediction based on codebook and model file
    logger.debug('Loading codebook and model file...')
    codebook, gtb = load_model_files(CODEBOOK_FILE, MODEL_FILE)
    logger.debug('The codebook parameters: \r\n' + pprint.pformat(codebook.get_params(), width=1, indent=4))
    logger.debug('The model parameters: \r\n' + pprint.pformat(gtb.get_params(), width=1, indent=4))
    logger.info('Start making predictions')
    sys.stdout.write("Making predictions...\n")

    missing_images_list = []
    predicted_images = 0
    for fname in flist:
        sys.stdout.write('>')
        sys.stdout.write('"' + fname + '";')
        feature = get_feature_vector(codebook, fname)
        if feature is None:
            # TODO: in an image list can be a path to a non-image file, like text or move etc. We should not to add such a files into the missing_images_list
            sys.stdout.write("\n")
            logger.warning('Can\'t make a prediction for file ' + fname)
            missing_images_list.append(fname)
            continue
        prediction = make_prediction(gtb, feature)
        if prediction is None:
            sys.stdout.write("\n")
            logger.warning('Can\'t make a prediction for file ' + fname)
            missing_images_list.append(fname)
            continue
        #curr_row = '"' + fname + '";' + str(prediction)
        predicted_images = predicted_images + 1
        #sys.stdout.write(curr_row + "\n")
        sys.stdout.write(str(prediction) + "\n")

    missing_images = len(missing_images_list)
    logger.info('Predictions made: ' + str(predicted_images))
    logger.info('Missing files: ' + str(missing_images))
    logger.info('Stop the application')

    sys.stdout.write("Done.\n")
    sys.stdout.write('Predictions made: ' + str(predicted_images) + "\n")
    if (missing_images > 0):
        sys.stderr.write('Could not predict ' + str(missing_images) + ' file(s):\n')
        for miss in missing_images_list:
            sys.stderr.write('"' + miss + '"\n')
