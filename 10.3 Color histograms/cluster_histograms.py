#!/bin/env python3

from sklearn.cluster import KMeans
from imutils import paths

from pyimagesearch.descriptors.labhistogram import LabHistogram

import numpy as np
import cv2
import argparse

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to the input dataset directory')
ap.add_argument('-k', '--clusters', type=int, default=2,
                help='# of clusters to generate')

args = vars(ap.parse_args())

# 8 bins per L,a,b channels -> 8x8x8 = 256d vector
desc = LabHistogram([8, 8, 8])
data = []

imagePaths = list(paths.list_images(args['dataset']))
imagePaths = np.array(sorted(imagePaths))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    hist = desc.describe(image)
    data.append(hist)

clt = KMeans(n_clusters=args['clusters'])
labels = clt.fit_predict(data)

for label in np.unique(labels):
    # numpy.where returns indices where the condition is true
    labelPaths = imagePaths[np.where(labels == label)]

    for (i, path) in enumerate(labelPaths):
        image = cv2.imread(path)
        cv2.imshow(f'Cluster {label + 1}, Image #{i + 1}', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
