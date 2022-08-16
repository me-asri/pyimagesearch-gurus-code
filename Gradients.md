# Gradients
## Summary
* Gradients are used for __edge detection__ in images.
* Formally, an image gradient is defined as a directional change in image intensity.
* Edge detection is the process of finding edges in an image, which reveals structural information regarding the objects in an image.
* An image gradient is a directional change in the intensity or color of an image.
* The gradient of the image is one of the fundamental building blocks in image processing. They can also be thought of as derivatives in 1D.
* In color images, gradients are calculated for each channel separately.
  * So at every individual pixel, we will have 6 numbers representing the gradient (Bx , Gx, Rx ) & (By ,Gy , Ry) for B, G and R channels respectively.
* At each pixel of the input _grayscale_ image, a gradient measures __the change in pixel intensity in a given direction__.
  * By estimating the direction or orientation along with the magnitude (i.e. how strong the change in direction is), we are able to detect regions of an image that look like edges.
* In practice, image gradients are estimated using kernels.
## Gradient Magnitude & Orientation
### Magnitude
* The gradient magnitude is used to measure _how strong_ the change in image intensity is.
* The gradient magnitude is a real-valued number that quantifies the _strength_ of the change in intensity.
* |G| = √(Gx^2 + Gy^2)
### Orientation
* Gradient orientation is used to determine in which direction the change in intensity is pointing.
* The gradient orientation will give us an angle that we can use to quantify the direction of the change.
* θ = arctan2(Gy / Gx)
## Sobel and Scharr kernels
* Since computing the magnitude and orientation for each and every pixel manually takes lot of time, let’s look at how we can _approximate_ the task of computing gradients using __kernels__, which will give us a tremendous boost in speed.
### Sobel
* Sobel method actually uses two kernels: one for detecting horizontal changes in X direction and the other for detecting vertical changes in Y direction
* Sobel method in OpenCV:
    ```
    gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)
    ```
### Scharr
* We can also use Scharr kernel instead of the Sobel kernel which may give us better approximations to the gradient
## Convolution
* We can apply a convolution to obtain the gradient images:
  * `Gx = I * Dx`
  * `Gy = I * Dy`
* Where `I` is the _image_ and `Dx` is our filter in the _x_ direction and `Dy` is our filter in the _y_ direction.
## Source
* [cvexplained](https://cvexplained.wordpress.com/2020/06/01/gradients/)