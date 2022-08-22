# Histogram of Oriented Gradients
## Summary
* HOG descriptors are mainly used in conjunction with a _Linear SVM_ for object detection. However, we can also use HOG descriptors for representing both *shape* and *texture*.
* HOG descriptors are mainly used to describe the shape and appearance of an object in an image. However since HOG captures local intensity gradients and edge directions, it also makes for a good texture descriptor.
* The cornerstone of the HOG descriptor is that **apperance of an object can be modeled by the distribution of _intensity gradients inside rectangular regions_ of an image**.
### Parameters
* The most important parameters for the HOG descriptors are the *orientations*, *pixels_per_cell* and *cells_per_block*.
  * These parameters control the dimensionality of the resulting feature vector.
### Cells
* Implementing HOG descriptor requires dividing the image into small connected regions called *cells* and then for each cell, computing a HOG for the pixels withing each cell. We can then accumulate these histograms accross multiple cells to form our feature vector.
### Block Normalization
* We can perform *block normalization* to improve performance.
* To perform block normalization we take groups of overlapping cells, concatenate their histograms, calculate a normalizing value and the contrast normalize the histogram.
* By normalizing over multiple overlapping blocks, the resulting descriptor is more robust to changes in illumination and shadowing.
## Stages of HOG Descriptor
1. Normalizing the image
2. Computing gradients in both x and y directions
3. Obtaining weighted votes in spatial and orientation cells
4. Contrast normalizing overlapping spatial cells
5. Collecting all HOGs to form the final feature vector
### Step 1: Normalizing the image
* The normalization step is optional, but in some cases this step can improve performance of the HOG descriptor.
* There are three main normalization methods we can consider:
  1. Gamma/power law normalization
  2. Square-root normalization
  3. Variance normalization
* In most cases it's best to start either with **no normalization** or **square-root normalization**.
  * Variance normalization is also worth considering but in most cases it'll perform similar to square-root normalization.
### Step 2: Gradient computation
* We can apply a convolution to obtain the gradient images:
  * `Gx = I * Dx`
  * `Gy = I * Dy`
* Where `I` is the _image_ and `Dx` is our filter in the _x_ direction and `Dy` is our filter in the _y_ direction.
* Now that we have the gradient images we can compute the final gradient magnitude representation of the image: `|G| = √(Gx^2 + Gy^2)`
* Finally the orientation of the gradient for each pixel in the input image can be computed by: `θ = arctan2(Gy / Gx)`