from sklearn.svm import LinearSVC

import argparse
import glob

import mahotas
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--training', required=True,
                help='Path to the dataset of textures')
ap.add_argument('-t', '--test', required=True,
                help='Path to the test images')
args = vars(ap.parse_args())

print('[INFO] Extracting features...')

data = []
labels = []

imagePaths = glob.glob(f"{args['training']}/*.png")
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # Heralick works with GLCM (Gray level co-occurance matrix)
    # - So we need a grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    texture = imagePath[(imagePath.rfind('/') + 1):].split('_')[0]

    # Calculate Heralick in 4 directions
    # - Remember GLCM can be calculated in 4 directions (each row is a direction)
    #   - Left to right, Top to bottom, Top-left to bottom-right, Top-right to bottom-left
    # Then calculate a mean of all directions (rows)
    # - Taking an average of the directions improves the robustness of the descriptor
    #   and improves rotation invariance (It's still not really robust against rotation)
    # - When using masks you may want to set ignore_zeros to True otherwise False
    features = mahotas.features.haralick(image).mean(axis=0)

    data.append(features)
    labels.append(texture)

print('[INFO] training model')

model = LinearSVC(C=10.0, random_state=42)
model.fit(data, labels)

print('[INFO calssifying]')

testImages = glob.glob(f"{args['test']}/*.png")
for testImage in testImages:
    image = cv2.imread(testImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = mahotas.features.haralick(gray).mean(axis=0)

    # Returns an array containing class label for each sample
    # - As we pass a single sample we just need the first element since the [0]
    pred = model.predict(features.reshape(1, -1))[0]

    # What's up with 'reshape(1, -1)'?
    # - LinearSVC's predict method expects a 2D array where each row is a sample feature
    #   but the 'features' array is 1D
    #   So reshape(1, -1) reshapes the array into a 2D array

    cv2.putText(image, pred, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
