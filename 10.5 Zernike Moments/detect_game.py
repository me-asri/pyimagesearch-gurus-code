from scipy.spatial import distance as dist

import numpy as np
import mahotas
import cv2


def describe_shapes(image: cv2.Mat) -> np.array:
    shapeFeatures = []

    # Convert image to graysacle
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to get rid of noise
    # See: https://cvexplained.wordpress.com/2020/05/26/smoothing-and-blurring/
    # Increasing the kernel size of the blur will intensify the blur effect
    # sigma variables define the smoothness
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    # cv2.threshold returns a tuple of 2 values
    # - the threshold value
    # - the thresholded image
    # ? How do we choose the threshold value?
    # - From the looks of it, there is no standard way of choosing one
    #   Could be case by case
    # - Some people like to use 255/3 = 85
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

    # What's "kernel"?
    # - An image kernel is a small matrix used to apply effects

    # What's up with erode and dilate?
    # Erosion shrinks shapes, it's mostly used to seperate parts of an object
    # Dilation expands shapes, it's mostly used to join seperate parts of an object
    # - Doing dilation after erosion gets rid of the white spots
    # - Doing erosion after dilation (this case) gets rid of the black spots (closing gaps)
    # * Morphological operations are only done on binary images
    # See: https://cvexplained.wordpress.com/2020/05/18/erosion/
    #      https://cvexplained.wordpress.com/2020/05/18/dilation/
    # ? How do we choose the kernel
    # Depends on the size of the holes we want to fill
    # See: https://stackoverflow.com/questions/67117928/how-to-decide-on-the-kernel-to-use-for-dilations-opencv-python
    # ? How do we choose the iterations?
    # I'm not exactly sure but the chances are it's chosen case by case
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)


distractor = cv2.imread('distractor.jpg')
describe_shapes(distractor)
