import csv
from time import gmtime, strftime
import numpy as np
import mahotas # for local binary patterns
import cv2

src_dir = 'd:/cheetah/cheetah2-train-preprocessed/'
csv_file = 'd:/cheetah/cheetah2-train/metadata.csv'
# radius=1, neigbors=8
#output_file = 'lbp-1-8.csv'

# Read list of image paths with labels from a csv file
def read_im_info(fname):
    with open(csv_file, 'rb') as fname:
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
    return  bad_list, good_list

# Extract histogram of lbp
def extract_lbp_hist(im_path, r, n):
    img = cv2.imread(im_path, 0)
    if None == img:
        print "Can't read image " + im_path + "!"
        return None
    return mahotas.features.lbp(img, r, n)

# Calculate image features and gather them in a nparray
def get_all_features(flist, r, n):
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " LBP parameters:"
    print " radius = " + str(r) + ", neigbors = " + str(n)
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Calculate images features..."
    output_file = 'lbp-' + str(r) + '-' + str(n) + 'csv'
    des = []
    i = 0
    out_csv = csv.writer(open(output_file, 'wb'))
    for fname in flist:
        img_path = src_dir + fname[0] + '.jpg'
        curr_img_features = extract_lbp_hist(img_path, r, n)
        if None == curr_img_features:
            print "\t Wrond descriptors' dimensions: " + img_path
            continue
        curr_row = [fname[2]] + [fname[0]] + [str(i) for i in curr_img_features]
        out_csv.writerow(curr_row)
        i = i + 1
        if 0 == i % 100:
            print "\t" + str(i) + " of " + str(len(flist))
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."
    return 1

# # flist - list of files
# # r - radius, n - number of neighbors
# def process_lbp(flist, r, n):
#     print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " LBP parameters:"
#     print " radius = " + str(r) + ", neigbors = " + str(n)
#     print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Calculate images features..."
#     all_features = get_all_features(flist, r, n)
#     print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Done."
#     output_file = 'lbp-' + str(r) + '-' + str(n) + 'csv'
#     with open(output_file, "wb") as f:
#         writer = csv.writer(f)
#         writer.writerows(all_features)
#     print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Features are saved."

if __name__ == '__main__':
    print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " Read files info..."
    flist = read_im_info(csv_file)
    get_all_features(flist, 1, 8)
    get_all_features(flist, 2, 16)
    # Too large file
    #process_lbp(flist[1:5], 3, 24)