# Hybrid images

With these programs hybrid images of two source images can be generated.

## Blending

The generated image consists of two halves of two images. It is created via three methods.

1. Direct blending (just cutting the images in half)
2. Laplacian blending (using laplace pyramid)
3. Laplacian blending with mask (using laplace pyramid for images and gauss pyramid for the mask)

### Examples

Apple             | Orange
:-------------------------:|:-------------------------:
![Apple](./blend/images/apple.jpg)  |  ![Orange](./blend/images/orange.jpg)

### Results

Method             | Image
:-------------------------:|:-------------------------:
Direct  |  ![Direct blending](./blend/examples/direct_blend.png)
Laplace w/o mask  |  ![Laplace blending w/o mask](./blend/examples/reconstructed.png)
Laplace w/ mask  |  ![Laplace blending w/ mask](./blend/examples/reconstructed-mask.png)


## Low-/High-frequency merge
The generated image consists of the low-frequency components of one image and the high-frequency components of the other one.
This results in seeing one or the other image depending on the distance it is looked upon.
If you are closer you see the picture from which the high frequencies have been extracted,
if you increase the distance you see the picture from which the low frequencies have been extracted.

### Examples

Original             |  Low-pass / High-pass (Sigma=10)
:-------------------------:|:-------------------------:
![Greyscale image 1](./low-high-pass/images/greyscale_1.png)  |  ![Low-pass greyscale image](./low-high-pass/examples/greyscale-low-pass.png)
![Greyscale image 2](./low-high-pass/images/greyscale_2.png)  |  ![High-pass greyscale image](./low-high-pass/examples/greyscale-high-pass.png)


Sigma             | Near             |  Far
:-------------------------:|:-------------------------:|:-------------------------:
Low: 10 High: 10  | <img src="./low-high-pass/examples/greyscale-hybrid.png" width="400">  |  <img src="./low-high-pass/examples/greyscale-hybrid.png" width="80">

Sigma             | Near             |  Far
:-------------------------:|:-------------------------:|:-------------------------:
Low: 10 High: 30  | <img src="./low-high-pass/examples/hybrid.png" width="400">  |  <img src="./low-high-pass/examples/hybrid.png" width="80">

Original             |  Low-pass / High-pass (Sigma=10)
:-------------------------:|:-------------------------:
![Color image 1](./low-high-pass/images/color_1.jpg)  |  ![Low-pass color image](./low-high-pass/examples/color-low-pass.png)
![Color image 2](./low-high-pass/images/color_2.jpg)  |  ![High-pass color image](./low-high-pass/examples/color-high-pass.png)

Near             |  Far
:-------------------------:|:-------------------------:
<img src="./low-high-pass/examples/color-hybrid.png" width="500">  |  <img src="./low-high-pass/examples/color-hybrid.png" width="50">
