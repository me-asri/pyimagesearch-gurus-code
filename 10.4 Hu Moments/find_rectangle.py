from sklearn.metrics.pairwise import pairwise_distances

import argparse
import glob
import numpy as np
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to dataset directory')
args = vars(ap.parse_args())

imagePaths = sorted(glob.glob(f"{args['dataset']}/*.jpg"))
data = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # We need a greyscale image for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold image
    # + Thresholding is one of the most common (and basic) segmentation techniques in computer vision
    #   and it allows us to separate the foreground (i.e. the objects that we are interested in) from the background of the image.
    # +  In general, we seek to convert a grayscale image to a binary image, where the pixels are either 0 or 255
    # For every pixel, a threshold value is applied.
    #   If the pixel value is smaller than the threshold value T, it is set to 0, otherwise it is set to a maximum value.
    # See: https://cvexplained.wordpress.com/2020/05/30/thresholding/
    #      https://datacarpentry.org/image-processing/07-thresholding/
    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Get the contour with max area
    c = max(cnts, key=cv2.contourArea)

    # Extract ROI
    (x, y, w, h) = cv2.boundingRect(c)
    # It's important to resize ROI so that features returned by Hu Moments
    #   are not influenced by the size of the object, *just the shape*
    roi = cv2.resize(thresh[y:(y + h), x:(x + w)], (50, 50))

    moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    data.append(moments)

# Distance matrix calculation
# - Calculates Euclidean distance between feature vectors
# There are 501 images so the distance matrix is 501x501

# NumPy 2D axes:
#   | - - - - - -> Axis 1
#   |   |   |   |
#   |------------
#   |   |   |   |
#   |------------
#  \ /
#  Axis 0

# Calculate distance matrix and then take sum of each row (axis 1)
# Why sum?
# When comparing circles to circles the distance should be small
# But when comparing a rectangle to a circle the distance should be large
D = pairwise_distances(data).sum(axis=1)
# argmax() returns the index of the element with largest value
# The row with the largest distance is the outlier
i = np.argmax(D)

image = cv2.imread(imagePaths[i])
print(f'Found square: {imagePaths[i]}')
cv2.imshow('Outlier', image)
cv2.waitKey(0)
