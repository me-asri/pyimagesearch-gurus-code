from skimage import feature, exposure
import cv2

logo = cv2.imread('training/audi/audi_01.png')
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True,
                            block_norm='L1', visualize=True)

# Rescale values into gray scale range
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# Output of exposure.rescale_intensity is an array of elements with the type float64
# But we expect pixels to be in the range of uint8
hogImage = hogImage.astype('uint8')

cv2.imshow('HOG Image', hogImage)
cv2.waitKey()
