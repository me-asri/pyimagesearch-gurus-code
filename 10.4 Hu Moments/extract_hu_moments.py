import cv2
import imutils

image = cv2.imread('planes.png')
# We need greyscale silhouettes for working with Hu Moments
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Incorrect way of doing it
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print(f'Original moments: {moments}')
cv2.imshow('Image', image)
cv2.waitKey(0)

# To correctly compute Hu Moments we have to
# 1. Find the contour of each plane
# 2. Extract the ROI (Region of Interest) surrounding the airplane
# 3. Compute Hu Moments for each of the ROIs *individually*
# See: https://cvexplained.wordpress.com/2020/06/03/finding-and-drawing-contours/

# What's the difference between contours and edges?
#   Edges outline the difference between the changes in color intensities
#   Contours outline the shape of an object. Contours are *closed* curves
#   You may obtain contours by connecting the edges to obtain a closed curve

# + findContours is *destructive* to the input image so pass a copy
# + Modes:
# RETR_EXTERNAL : Retrieves only the outer contours.
#   This is the fastest mode.
# RETR_LIST : Retrieves all of the contours without establishing any hierarchical relationships.
#   So you won’t know if one contour is nested inside another.
# RETR_CCOMP : Retrieves all of the contours and organizes them into a two-level hierarchy.
#   At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
# RETR_TREE : Retrieves all of the contours and reconstructs a full hierarchy of nested contours. This is slower but detailed.
# + A contour is simply list of points that form boundary of the shape.
#   One way is, to store all the points representing the boundary,
#   but it is wasteful to store hundreds of points for simple shapes like a triangle or a quad.
#   For a triangle, 3 points are enough and for a quad 4 points are enough.
#   This flag helps us choose the level of approximation
# CHAIN_APPROX_NONE : No approximation is used and all points are returned.
# CHAIN_APPROX_SIMPLE : This simple approximation algorithm that works well when the shapes are polygons.
#   It will return 4 points for a quad and 3 points for a triangle and so on and so forth.
# CHAIN_APPROX_TC89_L1 : This is a more accurate approximation algorithms.
#   This should be used when the shapes curved and are not simple polygons.
# CHAIN_APPROX_TC89_KCOS : This is computationally more expensive and slightly more accurate than CHAIN_APPROX_TC89_L1 algorithm.
#   This should be used when the shapes are curved and are not simple polygons.
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# What's up with the grab_countors function above?
# You see, the findContours function returns a *tuple of values*
# The problem with the returning tuple is that it is in a different format for OpenCV 2.4,
#   OpenCV 3, OpenCV 3.4, OpenCV 4.0.0-pre, OpenCV 4.0.0-alpha, and OpenCV 4.0.0
# To accommodate all these changes across different versions,
#   we will use imutils package to grab the contours.

for i, c in enumerate(cnts):
    # Extract the ROI from the image and compute the Hu Moments feature
    # - boundingRect() returns an approximate rectangle around shape
    # x, y are the starting points and w, h are the width and height of the rect.
    x, y, w, h = cv2.boundingRect(c)
    # Next we have to crop the ROI using NumPy slices
    # - image[y_start:y_end, x_start:x_end]
    # See: https://numpy.org/doc/stable/user/quickstart.html#the-basics
    roi = image[y:(y + h), x:(x + w)]
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()

    print(f'Moments for plance #{i + 1}: {moments}')
    cv2.imshow(f'ROI #{i + 1}', roi)
    cv2.waitKey(0)
