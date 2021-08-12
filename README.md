[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![PyPI](https://img.shields.io/pypi/v/a) [![Downloads](https://pepy.tech/badge/seg-crf)](https://pepy.tech/project/seg-crf) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FConditional-Random-Fields-CRF&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Fully Connected CRF

This repo implements CRF as described in Deeplab paper it takes about 0.2 seconds per image. Following image is taken form **DeepLab** paper

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img1.png)

## Requirements

```
Python <= 3.6
pydensecrf
cv2
matplotlib
gray2color
```
## Installation
via [PyPi](https://pypi.org/project/seg-crf/) or
```
pip install seg-crf
```

if you get error during installation due to `pydensecrf` then follow this [link](https://github.com/lucasb-eyer/pydensecrf) to resolve it or just type,

```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
## Usage

```python

from seg_crf import Seg_CRF

img_path='D:/Anaconda/Image_analysis/cat.png'
model_op_path='D:/Anaconda/Image_analysis/mask.png'

crf = Seg_CRF(img_path, model_op_path, 2, img_w=1024, img_h=512, clr_op=True, pallet2use ='cityscape')

gray, rgb = crf.start()
plt.imshow(rgb)

```
It takes following inputs.(see dir `sample_data` for sample masks) `gt` are just groundtruths they are not used during caculation

```
        ⚠ Zero pixels are consdered background
        img_path : path to an image, 
                        Format [H, W, 3]; values ranging from [0, 255]
        model_op_path : path model output of the same input image.
                        Format [H, W]; values ranging from [0, num_of_classes]
        num_of_classes : number of classes in a dataset e.g. in cityscape has 30 classes
        clr_op : color the output or not a bool
        pallet2use : see https://pypi.org/project/gray2color/ for details
        img_w : for resizing image and mask to same size default is 1024
        img_h : for resizing image and mask to same size default is 512
        apperance_kernel : The PairwiseBilateral term in CRF a list of values in order [sxy, srgb, compat]  
                            default values are [8, 164, 100]
        spatial_kernel : The PairwiseGaussian term in CRF a list of values in order [sxy, compat]  
                            default values are [3, 10]
```
More about spatial and apperance kernel below.
## Why CRF?

CRF’s are used for smoothing the noisy segmentation maps. See image below.

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img2.png)

## Types of CRF

### Grid CRF

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img4.png)

### Fully connected CRF
This is the one implemented in this repo.

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img5.png)

## Some mathematical Background

FC CRF consists of two Guassian Kernels one is called appearance kernel and other is called spatioal kernel. The spatial kernel is used of controlling the smoothness of the segmented regions.
and the appearace kernel controls which regions of segemneted image should be combined to one after lookin at origina input image.

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img3.png)

#### You can control the parameters of these kernels as follwoing
```
# inside the CRF function and class change;
#        1. increasing sxy puts more focus on changing intensity
#        2. if we increase srgb the pixels in close proximity will be assigned one class 
#           (high value will cause the erosion like effect at boundaries)
```
## Appearance and Spatial Kernel

```python
# Default Values are
apperance_kernel = [8, 164, 100] # PairwiseBilateral [sxy, srgb, compat]  
spatial_kernel = [3, 10]         # PairwiseGaussian  [sxy, compat] 

# or if you want to to specify seprately for each XY direction and RGB color channel then

apperance_kernel = [(1.5, 1.5), (64, 64, 64), 100] # PairwiseBilateral [sxy, srgb, compat]  
spatial_kernel = [(0.5, 0.5), 10]                  # PairwiseGaussian  [sxy, compat] 
# Use like
crf = Seg_CRF(img_path, model_op_path, 2, img_w=1024, img_h=512,
                 apperance_kernel=apperance_kernel, spatial_kernel=spatial_kernel,
                 clr_op=True, pallet2use ='cityscape')

gray, rgb = crf.start()
```
## FC-CRF in Machine Learning Pipeling

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img6.png)


## Visual Results 
For binar and multiclass segementation

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img7.png)
