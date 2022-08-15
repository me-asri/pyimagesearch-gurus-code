from skimage import feature

import numpy as np
import cv2


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius) -> None:
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # Create LBP image which has the same size as the input image
        # - LBP expects a grayscale image
        #   and returns a NumPy array containing the LBP image
        lbp = feature.local_binary_pattern(
            image, self.numPoints, self.radius, method='uniform')

        # Which LBP method to use?
        # - The 'default' method is grey-scale invariant however it's not rotation invariant
        # - The 'uniform' method is both grey-scale and rotation invariant
        # So, the 'uniform' method is preferred
        # - Assuming p=24 we get 25 invariant prototypes
        #   Along with an extra dimension for all patterns that are not unifrm
        #   yielding a total of 26 unique possible value

        # Create a histogram from the LBP image
        # - numpy.histogram() returns a tuple containing the histogram and bin_edges
        # The returned histogram is 26-d, an integer count for each prototype
        (hist, _) = np.histogram(lbp.ravel(), bins=range(
            0, self.numPoints + 3), range=range(0, self.numPoints + 2))

        # What's up with ravel()?
        # - ravel() takes an array and flattens it
        #   See: https://www.sharpsightlabs.com/blog/numpy-ravel/
        # How is it any different any than flatten()?
        # - ravel() can be called on any object that can successfully be parsed
        # - ravel() returns a *view* of the original array in a different shape
        #   thus modifying the returned array *will* modify the original array
        #   however flatten() always makes a copy

        # Why is the number of bins range(0, self.numPoints + 3)?
        # - Because there are p (self.numPoints) + 2 total patterns
        #   as last parameter of range is exclusive we have to use p + 3
        # Why is the range range(0, self.numPoints + 2)?
        # - TODO

        # Normalize the histogram such that it sums to 1
        # - We can't divide int64 by a float so we have to cast beforehand
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        # What's 'eps'?
        # - We use a very small number called epsilon
        #   to prevent division by zero in the case the histogram has zero entries

        return hist
