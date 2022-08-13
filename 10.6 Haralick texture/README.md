# Haralick texture
## Summary
* Haralick texture features are used to describe the texture of an image.
* Haralick features are derived from the Gray Level Co-occurence Matrix (GLCM). This matrix records how many times two gray-level pixels adjacent to each other appear in an image.
* Haralick propses 13 values that can be extracted from GLCM to quantify texture.
* GLCM can be calculated in 4 directions: Left to right, Top to bottom, Top-left to bottom-right, Top-right to bottom-left
* Taking an average of these directions to form a final feature vector of 13-dimensionality to make the feature vector more robust in changes in rotation.
* Haralick texture features are extracted from the entire image. To describe a part of an image we have to extract the ROI first.
## Pros and Cons
### Pros
* Very fast to compute
* Low dimensional (requires less space to store feature vector)
* No parameters to tune
### Cons
* Not very robust to changes in rotation
* Very sensitive to noise (a small change in grayscale image can dramatically affect the GLCM and thus the Haralick feature vector)
* Similar to Hu Moments, basic statistics are often not discriminative enough to distinguish between many different kinds of textures
