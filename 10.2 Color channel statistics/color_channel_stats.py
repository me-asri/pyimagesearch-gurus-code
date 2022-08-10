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
    # concatenate() basically joins a sequence of arrays
    # flatten() "collapses" a multi dimensional array into one dimension
    # see: https://www.w3resource.com/numpy/manipulation/index.php
    features = np.concatenate([means, stds]).flatten()
    index[filename] = features

# Query image for comparison
query = cv2.imread(imagePaths[0])
cv2.imshow(f"Query ({imagePaths[0]}", query)

keys = sorted(index.keys())
for i, k in enumerate(keys):
    # Ignore first image
    if k == "trex_01.png":
        continue

    # Load current image and calculate Euclidean distance between current and query image
    image = cv2.imread(imagePaths[i])
    d = dist.euclidean(index["trex_01.png"], index[k])

    cv2.putText(image, "%.2f" % (d), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(k, image)

# Wait for key press indefinitely
cv2.waitKey(0)
