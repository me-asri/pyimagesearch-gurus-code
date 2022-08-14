from skimage import feature

import numpy as np
import cv2


class LocalBinaryPatterns:
    def __init__(self, numPoints: int, radius: float) -> None:
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image: cv2.Mat, eps: float = 1e-7):
        lbp = feature.local_binary_pattern(
            image, self.numPoints, self.radius, method='uniform')

        # Which LBP method to use?
        # - The 'default' method is grey-scale invariant however it's not rotation invariant
        # - The 'uniform' method is both grey-scale and rotation invariant.
        # So, the 'uniform' method is preferred
