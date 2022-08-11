import numpy as np
import argparse
import uuid
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='path to the output directory')
ap.add_argument('-n', '--num-images', type=int, default=500,
                help='# of disctrator images to generate')
args = vars(ap.parse_args())

for i in range(0, args['num_images']):
    # Create empty black image (filled with zeros)
    image = np.zeros((500, 500, 3), dtype='uint8')

    # Generate 2 random integer ('int0') values (x and y) between 105 (inc) and 395 (exc) as x, y positions
    # int0 is an alias for intp which is equals to ssize_t in C
    # See: https://docs.scipy.org/doc/numpy-1.10.0/user/basics.types.html
    (x, y) = np.random.uniform(low=105, high=395, size=(2,)).astype('int0')
    # Generate a random integer value as circle radius
    r = np.random.uniform(low=25, high=100, size=(1,)).astype('int0')[0]

    # Generate 3 random values (B,G,R) as color for circus
    color = np.random.uniform(low=0, high=255, size=(3,)).astype('int0')
    # Convert color array to 'int' tuple (cv2.circle() expects a tuple of 3 ints as color)
    # - Python's 'int' is different than NumPy's 'numpy.int64'
    # - cv2.circle expects a tuple of 'int's not 'numpy.int64'
    color = tuple(map(int, color))

    # Draw the circle
    # A thickness of -1 creates a filled in shape
    # See: https://cvexplained.wordpress.com/2020/04/26/drawing/
    cv2.circle(image, (x, y), r, color, -1)

    cv2.imwrite(f"{args['output']}/{uuid.uuid4()}.jpg", image)

# Allocate memory for rectangle image
image = np.zeros((500, 500, 3), dtype='uint8')
topLeft = np.random.uniform(low=25, high=225, size=(2,)).astype('int0')
botRight = np.random.uniform(low=250, high=400, size=(2,)).astype('int0')

# Random BGR tuple
color = np.random.uniform(low=0, high=255, size=(3,)).astype('int0')
color = tuple(map(int, color))

cv2.rectangle(image, tuple(topLeft), tuple(botRight), color, -1)
cv2.imwrite(f"{args['output']}/{uuid.uuid4()}.jpg", image)
