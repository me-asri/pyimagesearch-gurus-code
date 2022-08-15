# Local Binary Patterns
## Summary
* Local Binary Patterns (LBPs) are a texture descriptor.
* Unlike Haralick texture features that compute a _global representation_  of texture, LBPs compute a _local representation_ of texture.
* In general, LBPs are substantially more power than Haralick. However it's computationally more prohibitive and expensive.
* LBP has been successfully used to perform robust face recognition.
## Extensions to LBP
* The original LBP algorithm handles 3x3 neighborhood. It's fast, efficient and able to catch small details but we can't capture any other scale
* To handle variable neighborhoods two parameters were introduced
  * p: Number of points in a circularly symmetric neighborhood to consider
  * r: raidus of the circle which allows us to account for different scales
* By increasing _p_ the more patterns we can encode but at the same time increasing the computational cost.
* By increasing _r_ we can capture larger texture details in image. However increasing _r_ without increasing _p_ we'll lose the descriminative power of the LBP descriptor
* In general we should increase or decrease both _r_ and _p_ together.
## Uniform patterns
* A LBP is considered to be uniform if it has at most two 0-1 or 1-0 transitions. For example, the pattern `00001000` (2 transitions) and `10000000` (1 transition) are both considered to be uniform patterns since they contain at most two 0-1 and 1-0 transitions. The pattern `01010010` on the other hand is not considered a uniform pattern since it has six 0-1 or 1-0 transitions.
* The number of uniform prototypes in a Local Binary Pattern is completely dependent on the number of points p. As the value of p increases, so will the dimensionality of your resulting histogram.
* Given the number of points p in the LBP there are *p + 1* **uniform patterns**. The final dimensionality of the histogram is thus *p + 2*, where **the added entry tabulates all patterns that are not uniform**.
## Histogram
* If we took all the LBP codes and constructed a histogram of them we would lose all spatial information similar to constructing a color histogram.
* But if we divide our image into blocks, extract LBPs for each block, and concatenate them together we're able to create a descriptor that encodes spatial information.
* Spatial encoding step is not necessary but for some tasks such as face recognition it's crucial.
## Colors
* LBP does not account for colors
* A few ways we can extend LBP to recognize colors:
  * Ranking the images first on texture, then re-ranking on color
  * Fusing both color and texture into single feature vector
  * Performing a search on both color and texture independently, and then merging the results together
## Choosing parameters
* Radius and point count has effect on the dimensionality of the feature vector and computational efficiency
* When using the `uniform` method the feature vector size will remain 25-d but computation times will increase
* The larger the number of points and radius, the slower extraction will be
* The author recommends starting off with _p=8_ and _r=1.0_ up to _p=24_ and _r=3.0_ then increasing the _radius_ to see if the accuracy improves
* The author also recommends using rotation invariant LBPs as they have substantially smaller feature vector size and are easier to compute a histogram for
## Pros & Cons
### Pros
* The original implementation of LBP is not rotation invariant however some extensions to it are (such as the uniform method in scikit package)
* Very good at characterizing the texture of an image
* Useful in face recognition
### Cons
* Can easily lead to large feature vectors if not careful
* Computationally prohibitive as _p_ and _r_ increase
## Sources
* [PyImageSearch](https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)