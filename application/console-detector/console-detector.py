#!/usr/bin/env python

import logging  # Tracking events that happen when the application runs
from time import gmtime, strftime  # Date-time for logging messages
import sys  # For an error message
import argparse  # Command line arguments
import pprint  # Pretty print some dictionaries (parameters of codebook and model)

# Reading file list from a directory or from a text file
from os import listdir
from os.path import isdir, exists, isfile, join

import cPickle  # Load codebook and model

import numpy as np  # Dealing with feature vectors

import csv  # Write result of prediction to a comma separated file

# Image processing, SURF features
import cv2
# LBP features
import mahotas

CODEBOOK_FILE = "./codebook-kmeans-500-500-50bins.cb"
GTB_MODEL_FILE = "./coarse-gtb.model"

logger = logging.getLogger('cheetah-detector')
logger.setLevel(logging.DEBUG)
# In the file we will log detail information
log_fname = strftime("%Y-%m-%d.log", gmtime())
fh = logging.FileHandler(log_fname)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Prints a message to stderr and exits
def error(message):
    sys.stderr.write(message + "\n")
    sys.exit(1)


# Extracts full file names of images from a directoty
# Test path: d:\cheetah_trials\gbm_detected_bad
def extract_image_paths(dir_path):
    flist = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return flist


# Check if all necessary files exist and load them.
# We need precalculated codebook and a model file
def load_model_files(codebook_file, gtb_model_file):
    if not exists(codebook_file):
        error("Error: can't find a codebook in the working directory")
        logger.error('Can\'t find a codebook in the working directory')
    if not exists(gtb_model_file):
        error("Error: can't find a model file in the working directory")
        logger.error('Can\'t find a model file in the working directory')

    with open(codebook_file, 'rb') as fid:
        codebook = cPickle.load(fid)
        logger.debug('Load a codebook')
    # TODO: check if the codebook is valid

    with open(gtb_model_file, 'rb') as fid:
        model = cPickle.load(fid)
        logger.debug('Load a model file')
    # TODO: check if the model is valid

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
def extract_img_feature(im_path, feature_name):
    img = cv2.imread(im_path, 0)
    if None == img:
        return None
    img = preprocess_img(img)
    if feature_name == 'sift':
        detector = cv2.SIFT()  # Not tested
    elif feature_name == 'surf':
        detector = cv2.SURF(700)
    elif feature_name == 'orb':
        detector = cv2.ORB(10)  # Not tested
    elif feature_name == 'brisk':
        detector = cv2.BRISK(50)  # Not tested
    else:
        print "Unknown image feature: " + feature_name
        logger.warning('Unknown image feature: ' + feature_name)
        return None
    kp, des = detector.detectAndCompute(img, None)
    return des


# Find BoW for an image
def get_bow(codebook, img_path, feature_name, cluster_count):
    img_features = extract_img_feature(img_path, feature_name)
    if None == img_features or codebook.cluster_centers_.shape[1] != img_features.shape[1]:
        return None
    pred = codebook.predict(img_features)
    hist, bin_edges = np.histogram(pred, bins=cluster_count, range=(1, cluster_count), normed=True)
    return hist


# Extract histogram of lbp
def extract_lbp_hist(im_path, r, n):
    img = cv2.imread(im_path, 0)
    if None == img:
        return None
    img = preprocess_img(img)
    return mahotas.features.lbp(img, r, n)


# Get feature vector for an image
def get_feature_vector(codebook, image_path):
    # TODO: made image preprocessing once
    bow = get_bow(codebook, image_path, 'surf', codebook.n_clusters)
    if (bow is None) or (len(bow) != codebook.n_clusters):
        logger.warning('Bad BoW descriptor for image ' + image_path)
        return None
    lbp = extract_lbp_hist(image_path, 1, 8)
    if lbp is None:
        logger.warning('Bad LBP descriptor for image ' + image_path)
        return None

    res = np.array(list(bow) + list(lbp)).astype(np.float32)
    return res


# Given a model file and a feature vector, make prediction
def make_prediction(model, feature):
    pred = model.predict(feature)
    # Dealing with values outside the range [0, 1]
    res = min(1, max(0, pred[0]))
    return res

# http://stackoverflow.com/a/3173331
def update_progress(progress):
    print '\r[{0}] {1}%'.format('#'*(progress/10), progress),

if __name__ == '__main__':

    logger.info('Start the application')
    logger.debug('Number of arguments: ' + str(len(sys.argv)))
    logger.debug('Argument list: ' + str(sys.argv))
    logger.debug('Codebook file name: ' + CODEBOOK_FILE)
    logger.debug('Model file name: ' + GTB_MODEL_FILE)

    input_source = ''
    output_file = ''

    parser = argparse.ArgumentParser(description='Recognizes images with spotted cats')
    parser.add_argument("input", \
                        help='Either path to a directory with images or a text file containing paths to images')
    parser.add_argument("output", \
                        help='Name of output file')

    # TODO: prints 'None' at the end of the help message, fix it
    # print parser.print_help()

    args = parser.parse_args()
    input_source = args.input
    output_file = args.output
    logger.debug("Input source: " + input_source)
    logger.debug("Output file: " + output_file)
    if isdir(input_source):
        flist = extract_image_paths(input_source)
    elif isfile(input_source):
        with open(input_source) as f:
            flist = f.readlines()
        flist = [x.strip('\n') for x in flist]
    else:
        logger.error('Unknown source of input data')
        error('Error: unknown source of input data')

    flist_len = len(flist)
    print "Images to process: " + str(flist_len)
    logger.debug('Number of files to process: ' + str(flist_len))

    # Read every image from the list, generate a feature vector
    # and make prediction based on codebook and model file
    logger.debug('Loading codebook and model file...')
    codebook, gtb = load_model_files(CODEBOOK_FILE, GTB_MODEL_FILE)
    logger.debug('The codebook parameters: \r\n' + pprint.pformat(codebook.get_params(), width=1, indent=4))
    logger.debug('The model parameters: \r\n' + pprint.pformat(gtb.get_params(), width=1, indent=4))
    logger.info('Start making predictions')
    print "Making predictions..."
    out_csv = csv.writer(open(output_file, 'wb'))
    missing_images = 0
    predicted_images = 0
    for fname in flist:
        feature = get_feature_vector(codebook, fname)
        if None == feature:
            logger.warning('Can\'t make a prediction for image ' + fname)
            missing_images = missing_images + 1
            continue
        prediction = make_prediction(gtb, feature)
        curr_row = [fname] + [str(prediction)]
        predicted_images = predicted_images + 1
        out_csv.writerow(curr_row)
        progress = 100*(missing_images + predicted_images) / flist_len
        update_progress(progress)

    logger.info('Predictions made: ' + str(predicted_images))
    logger.info('Missing images: ' + str(missing_images))
    logger.info('Stop the application')

    print "\nDone."