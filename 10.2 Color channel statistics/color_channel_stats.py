#!/bin/env python3

from scipy.spatial import distance as dist
from imutils import paths

import numpy as np
import cv2

imagePaths = sorted(list(paths.list_images("dinos")))

# Dict containing image filename and feature vectors
index = {}

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    filename = imagePath[(imagePath.rfind("/") + 1):]

    # Extract mean and standard deviation from each channel of RGB image
    # then update the index dict
    means, stds = cv2.meanStdDev(image)
    print(means)
    print(stds)
    # concatenate() basically joins a sequence of arrays
    # flatten() "collapses" a multi dimensional array into one dimension
    # see: https://www.w3resource.com/numpy/manipulation/index.php
    print(np.concatenate([means, stds]).flatten())
    features = np.concatenate([means, stds]).flatten()
    index[filename] = features
