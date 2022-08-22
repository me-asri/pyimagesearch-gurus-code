# Histogram of Oriented Gradients
## Summary
* HOG descriptors are mainly used in conjunction with a _Linear SVM_ for object detection. However, we can also use HOG descriptors for representing both *shape* and *texture*.
* HOG descriptors are mainly used to describe the shape and appearance of an object in an image. However since HOG captures local intensity gradients and edge directions, it also makes for a good texture descriptor.
* The cornerstone of the HOG descriptor is that **apperance of an object can be modeled by the distribution of _intensity gradients inside rectangular regions_ of an image**.
### Parameters
* The most important parameters for the HOG descriptors are the *orientations*, *pixels_per_cell* and *cells_per_block*.
  * These parameters control the dimensionality of the resulting feature vector.
* Parameters should be obtained by experimentation and examining the accuracy of the classifier.
### Cells
* Implementing HOG descriptor requires dividing the image into small connected regions called *cells* and then for each cell, computing a HOG for the pixels withing each cell. We can then accumulate these histograms accross multiple cells to form our feature vector.
### Block Normalization
* We can perform *block normalization* to improve performance.
* To perform block normalization we take groups of overlapping cells, concatenate their histograms, calculate a normalizing value and the contrast normalize the histogram.
* By normalizing over multiple overlapping blocks, the resulting descriptor is more robust to changes in illumination and shadowing.
### Image Size
* Images with different widths and heights result in **HOG feature vector of different sizes.**
  * When extracting HOG features from a dataset of images you'll want to define a *canonical, well known size* that each image will resized to.
  * This means throwing away the aspect ratio which we *usually* want to avoid but in this case we're happy to do this because:
    1. It ensures that each image in our dataset is described in a consistent manner
    2. Each feature vector is of the same dimensionality
## Stages of HOG Descriptor
1. Normalizing the image
2. Computing gradients in both x and y directions
3. Obtaining weighted votes in spatial and orientation cells
4. Contrast normalizing overlapping spatial cells
5. Collecting all HOGs to form the final feature vector
### Step 1: Normalizing the image
* The normalization step is **optional**, but in some cases this step can improve performance of the HOG descriptor.
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
### Step 3: Weighted votes in each cell
* A *cell* is a rectangular region defined by the number of pixels that belong in each cell.
  * For example if we had a 128x128 image and defined `pixels_per_cell` as 4x4, we would have 32x32 = 1024 cells.
* Now, for each of the cells in the image, we need to construct HOG using our gradient magnitude `|G|` and orientation `θ`.
* The number of _orientations_ control the number of _bins_ in the resulting histogram.
* The gradient angle is either within the range `[0, 180]` (unsigned) or `[0, 360]` (signed)
  * It's preferable to use gradients in the range `[0, 180]` with _orientations_ somewhere in the range `[9, 12]`. But depending on your application using signed gradients over unsigned gradients can improve accuracy.
* Finally, each pixel contributes a *weighted vote* to the histogram - **_The weight of the vote is simply the gradient magnitude |G| at the given pixel._**
* We could collect and concatenate each of these histograms to form our final feature vector. But it's beneficial to apply block normalization.
### Step 4: Contrast normalization over blocks
* To account for changes in *illumination* and *contrast*, we can normalize the gradient values *locally*.
  * This requires grouping the *cells* into larger *blocks*. It's common for these blocks to *overlap*, **_meaning that each cell contributes to the final feature vector more than once._**
* The number of blocks are rectangular however our units are no longer pixels - they are cells.
* Using either 2x2 or 3x3 `cells_per_block` obtains reasonable accuracy in most cases.
* For each of the cells in the current block we concatenate their corresponding gradient histograms. Followed by either `L1` or `L2` normalizing the entire concatenated feature vector.
* Performing this type of normalization implies that each of the cells will be represented in the final feature vector multiple times but normalized by a different value.
* After all blocks are normalized we take the resulting histograms, concatenate them, and treat them as our final feature vector.