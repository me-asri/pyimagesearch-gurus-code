from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure, feature
from imutils import paths

import argparse
import imutils
import cv2

# Paramaters for HOG descriptor
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (10, 10),
    'cells_per_block': (2, 2),
    'transform_sqrt': True,
    'block_norm': 'L1'
}

CANONICAL_SIZE = (200, 100)

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--training', required=True,
                help='Path to the logos training dataset')
ap.add_argument('-t', '--test', required=True,
                help='Path to the test dataset')

args = vars(ap.parse_args())

print('[INFO] extracting features...')

data = []
labels = []
for imagePath in paths.list_images(args['training']):
    make = imagePath.split('/')[-2]

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply auto Canny edge detection and return the edged image
    # The edged image contains the outline of the logo
    edged = imutils.auto_canny(gray)

    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the biggest contour
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the bounding box
    (x, y, w, h) = cv2.boundingRect(c)
    # Extract the ROI
    logo = gray[y:(y + h), x:(x + w)]
    logo = cv2.resize(logo, CANONICAL_SIZE)

    H = feature.hog(logo, **HOG_PARAMS)

    # HOG parameters were obtained by experimentation and

    data.append(H)
    labels.append(make)

print('[INFO] training classifier')
# Train the nearest neighbors classifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)

print('[INFO] evaluating...')

for (i, imagePath) in enumerate(paths.list_images(args['test'])):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo = cv2.resize(gray, CANONICAL_SIZE)

    (H, hogImage) = feature.hog(logo, visualize=True, **HOG_PARAMS)

    pred = model.predict(H.reshape(1, -1))[0]

    # What's up with reshape(1, -1)?
    # - reshape(1, -1) reshapes the array into single row with unknown count of columns,
    # - reshape(1, -1) returns a 2D array while flatten returns a 1D array
    # - predict() requires a 2D array where each row is a feature

    # What's up with the [0]?
    # - feature.hog() returns an array containing class label for each feature
    # - As we pass a single sample we just need the first element hence the [0]

    # Visualize the HOG image
    # - Visualizing the HOG image helps us debug whether images get quantified adequately or not
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype('uint8')
    cv2.imshow(f'HOG Image #{i + 1}', hogImage)

    cv2.putText(image, pred.title(), (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.imshow(f'Test Image #{i + 1}', image)
    cv2.waitKey(0)
