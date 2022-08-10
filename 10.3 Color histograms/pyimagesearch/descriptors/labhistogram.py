import cv2
import imutils
import numpy as np


class LabHistogram:
    # L*a*b* (Lab) color space
    # L*: lightness
    # a*: red/green
    # b*: blue/yellow
    # Lab color space is based on how humans preceive colors
    # See: https://cvexplained.wordpress.com/2020/05/26/lighting-and-color-spaces/

    # A histogram groups data into "bins" of equal width
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image, mask=None):
        # By default OpenCV works with BGR ordering
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        # See: https://cvexplained.wordpress.com/2020/06/07/histograms/
        hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins,
                            [0, 256, 0, 256, 0, 256])

        # Calculating histogram for OpenCV 2.4
        if imutils.is_cv2():
            # Normalizing ensures that width & height has no impact on the histograms
            hist = cv2.normalize(hist).flatten()
        # Calculating histogram for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()

        return hist
