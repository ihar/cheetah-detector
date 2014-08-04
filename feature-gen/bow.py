# Bag Of Words for image classification

import csv
import numpy as np
import cv2
from sklearn.cluster import KMeans
from time import gmtime, strftime
import cPickle #to save k-means model to file and read it later

# Where the whole image set is
SRC_DIR = 'd:/cheetah_trials/cheetah/cheetah2-train/'
# Description of the files with the class labels
CSV_FILE = 'd:/cheetah_trials/cheetah/metadata-cleaned_up.csv'

# How many pockets should be in BOW histogram
# Dimension of an image feature vector
CLUSTER_COUNT = 50

# Name of feature to extract from an image
# Can be any from the list: {sift, surf, orb, brisk}
# but only surf are used in the competition
FEATURE = 'surf'

# Should the script generate a codebook (centers of CLUSTER_COUNT clusters)
# for future use in feature vector generating?
GENERATE_CODEBOOK = True

# Should the script generate feature vector for each image from the image set
# and save it to a file for future use in classification?
GENERATE_BOW = True

# Constructed using per GOOD_LENGTH images from positive class
# and BAD_LENGTH images from negative class

# Because of memory issues, we could not use all images for generating a codebook.
# It was decided to generate the codebook based on a subset of the whole data set.
GOOD_LENGTH = 500 # How many images labeled as including a spotted cat should we take
BAD_LENGTH = 500  # How many images labeled as not including any spotted cat should we take

# Save codebook to a binary file, save features to comma-separated text file
CODEBOOK_FNAME = "codebook-kmeans-" + str(GOOD_LENGTH) + "-"+str(BAD_LENGTH) + "-" + str(CLUSTER_COUNT) + "bins.cb"
BOW_FNAME = "bow-" + str(GOOD_LENGTH) + "-" + str(BAD_LENGTH) + "-" + str(CLUSTER_COUNT) + ".csv"

# Read list of image paths with labels from a csv file
def read_im_info(fname):
    with open(fname, 'rb') as fname:
        data = list(tuple(rec) for rec in csv.reader(fname, delimiter=','))
    # Remove header of the csv file from the list
    del data[0]
    return data

# Preprocess image
def preprocess_img(image_in):
    # Preprocessing step: histogram equalization and reduced image size proportionally to width 300 pixels
    img = cv2.equalizeHist(image_in)
    scale_factor = 300.0/img.shape[:2][1]
    small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    return small_img

# Split data into two parts, with or without a big cat
# docID,daytime,target
# Returns two lists with image paths
def split_image_info(data):
    good_list = []
    bad_list = []
    for item in data:
        if '0' == item[2]:
            bad_list.append(item)
        else:
            good_list.append(item)
    return  bad_list, good_list

# Extract image features
# feature_name = {sift, surf, orb, brisk}
def extract_img_feature(im_path, feature_name):
    detector = None
    img = cv2.imread(im_path, 0)
    if None == img:
        print "Can't read image " + im_path + "!"
        return None
    img = preprocess_img(img)
    if feature_name == 'sift':
        detector = cv2.SIFT() # Not tested
    elif feature_name == 'surf':
        detector = cv2.SURF(700)
    elif feature_name == 'orb':
        detector = cv2.ORB(10) # Not tested
    elif feature_name == 'brisk':
        detector = cv2.BRISK(50) # Not tested
    else:
        print "Unknown image feature: " + feature_name
        return None
    kp, des = detector.detectAndCompute(img, None)
    return des

# Calculate image features and gather them in a nparray
def get_all_features(src_dir, flist, feature_name):
    des = []
    i = 0
    for fname in flist:
        img_path = src_dir + fname[0]
        curr_img_features = extract_img_feature(img_path, feature_name)
        if None == curr_img_features or 128 != curr_img_features.shape[1]:
            # print "\t Wrond descriptors' dimensions: " + img_path
            continue
        des.append(curr_img_features)
        i = i + 1
        if 0 == i % 100:
            print "\t" + str(i) + " of " + str(len(flist))
    return np.concatenate(des)

# Find a clusters' centers and save kmeans model to a file
def construct_codebook_and_save_it(features, cluster_count, fname):
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Start building a codebook..."
    r, c = features.shape[:]
    print "\t Feature space: " + str(r) + " x " + str(c)
    kmeans = KMeans(cluster_count)
    kmeans.fit(features)
    with open(fname, 'wb') as fid:
        cPickle.dump(kmeans, fid)
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " The codebook is ready and saved to file."

# read codebook from a file
# returns trained k-means object
def read_codebook_from_file(fname):
    with open(fname, 'rb') as fid:
        kmeans_loaded = cPickle.load(fid)
    return kmeans_loaded

# Find BoW for an image
def get_bow(model, fname, feature_name, src_dir):
    img_path = src_dir + fname + '.jpg'
    img_features = extract_img_feature(img_path, feature_name)
    if None == img_features or 128 != img_features.shape[1]:
        return None
    pred = model.predict(img_features)
    hist, bin_edges =  np.histogram(pred, bins=CLUSTER_COUNT, range=(1, CLUSTER_COUNT), normed=True)
    return hist

if __name__ == '__main__':

    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Read files info..."
    flist = read_im_info(CSV_FILE)
    if GENERATE_CODEBOOK:
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Split the file list into two parts..."
        good_list = []
        bad_list = []
        for item in flist:
            if '0' == item[2]:
                bad_list.append(item)
            else:
                good_list.append(item)

        print "\t Good list size: " + str(len(good_list))
        print "\t Bad list size: " + str(len(bad_list))

        # Memory error when use whole size of good samples and the same size of bad sample
        # File "bow.py", line 78, in get_all_features
        # return np.concatenate(des)
        # MemoryError
        # Reduce again the two data sets to size of 500
        # Memory error when make k-means model. Reduce further to 250
        common_list = good_list[0:GOOD_LENGTH] + bad_list[0:BAD_LENGTH]
        print "\t Common list size: " + str(len(common_list))

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Calculate images features..."
        all_features = get_all_features(SRC_DIR, common_list, FEATURE)
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."

        construct_codebook_and_save_it(all_features, CLUSTER_COUNT, CODEBOOK_FNAME)

    if GENERATE_BOW:
        kmeans = read_codebook_from_file(CODEBOOK_FNAME)

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Generating BoW for the image collection..."
        img_counter = 0
        out_csv = csv.writer(open(BOW_FNAME, 'wb'))
        for fline in flist:
            img_counter = img_counter + 1
            hist = get_bow(kmeans, fline[0], FEATURE, SRC_DIR)
            # If can't find any keypoint on an image, made histogram of zeroes
            if (hist is None) or (len(hist) != CLUSTER_COUNT):
                hist = [0]*CLUSTER_COUNT
            curr_row = [fline[2]] + [fline[0]] + [str(i) for i in hist]
            out_csv.writerow(curr_row)
            if img_counter % 1000 == 0:
                print "\t " + str(img_counter)
        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."

