# Local binary patterns for image classification

import csv
from time import gmtime, strftime
import mahotas
import cv2

SRC_DIR = 'd:/cheetah_trials/cheetah/cheetah2-train/'
#SRC_DIR = 'd:/cheetah/cheetah2-train/'
# Description of the files with the class labels
CSV_FILE = 'd:/cheetah_trials/cheetah/metadata-cleaned_up.csv'
#CSV_FILE = 'd:/cheetah/metadata-cleaned_up.csv'

PATTERN_RADIUS = 1
PATTERN_NEIGHBORS = 8

# Read list of image paths with labels from a csv file
def read_im_info(csv_fname):
    with open(csv_fname, 'rb') as fname:
        data = list(tuple(rec) for rec in csv.reader(fname, delimiter=','))
    # Remove header of the csv file from the list
    del data[0]
    return data

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
    return bad_list, good_list

# Preprocess image
def preprocess_img(image_in):
    # Preprocessing step: histogram equalization and reduced image size proportionally to width 300 pixels
    img = cv2.equalizeHist(image_in)
    scale_factor = 300.0/img.shape[:2][1]
    small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    return small_img

# Extract histogram of lbp
def extract_lbp_hist(im_path, r, n):
    img = cv2.imread(im_path, 0)
    if None == img:
        print "Can't read image " + im_path + "!"
        return None
    img = preprocess_img(img)
    return mahotas.features.lbp(img, r, n)

# Calculate image features and gather them in a nparray
def get_all_features(flist, r, n, src_dir):
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " LBP parameters:"
    print " radius = " + str(r) + ", neighbors = " + str(n)
    output_file = 'lbp-' + str(r) + '-' + str(n) + '.csv'
    i = 0
    out_csv = csv.writer(open(output_file, 'wb'))
    len_flist = len(flist)
    for fname in flist:
        img_path = src_dir + fname[0]
        curr_img_features = extract_lbp_hist(img_path, r, n)
        if None == curr_img_features:
            print "\t Wrond descriptors' dimensions: " + img_path
            continue
        curr_row = [fname[2]] + [fname[0]] + [str(k) for k in curr_img_features]
        out_csv.writerow(curr_row)
        i = i + 1
        if 0 == i % 500:
            percent = 100.0*i/len_flist
            print "\t" + str(i) + " of " + str(len(flist)) + ', ' + str(round(percent, 2)) + '%'
    return 1

if __name__ == '__main__':
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Read files info..."
    flist = read_im_info(CSV_FILE)
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Calculating LBP..."
    get_all_features(flist, PATTERN_RADIUS, PATTERN_NEIGHBORS, SRC_DIR)
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."
