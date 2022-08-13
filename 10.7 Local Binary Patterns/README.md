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
* Given the number of points _p_ in the LBP, there are _p + 1_ **uniform patterns**.
## Histogram
* If we took all the LBP codes and constructed a histogram of them we would lose all spatial information similar to constructing a color histogram.
* But if we divide our image into blocks, extract LBPs for each block, and concatenate them together we're able to create a descriptor that encodes spatial information.
* Spatial encoding step is not necessary but for some tasks such as face recognition it's crucial.
