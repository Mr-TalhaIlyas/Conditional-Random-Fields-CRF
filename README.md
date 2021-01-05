# Fully Connected CRF

This repo implements CRF as described in Deeplab paper it takes about 0.2 seconds per image. Following image is taken form **DeepLab** paper

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img1.png)

It takes following inputs.

```
    img_path : path to an image, 
                    Format [H, W, 3]; values ranging from [0, 255]
    model_op_path : path model output of the same input image.
                    Format [H, W]; values ranging from [0, num_of_classes]
    gt_path : path to ground truth of the same image. 
                Format [H, W]; values ranging from [0, num_of_classes]
    num_of_classes : number of classes in a dataset e.g. in cityscape has 30 classes
    clr_op : color the output or not a bool
    pallet: a [1 x no of classes x 3] array containing the RGB values of classes, type float32
```
## Requirements

* Python 3.6
* pydensecrf
* cv2
* tensorflow
* matplotlib

## Why CRF?

CRF’s are used for smoothing the noisy segmentation maps. See image below.

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img2.png)

##Types of CRF

### Grid CRF

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img4.png)

### Fully connected CRF
This is the one implemented in this repo.

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img5.png)

## Some mathematical Background

FC CRF consists of two Guassian Kernels one is called appearance kernel and other is called spatioal kernel. The spatial kernel is used of controlling the smoothness of the segmented regions.
and the appearace kernel controls which regions of segemneted image should be combined to one after lookin at origina input image.

![alt text](https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF/blob/master/screens/img3.png)

####You can control the parameters of these kernels as follwoing
```
# inside the CRF function and class change;
#        1. increasing sxy puts more focus on changing intensity
#        2. if we increase srgb the pixels in close proximity will be assigned one class 
#           (high value will cause the erosion like effect at boundaries)
```
## FC-CRF in Machine Learning Pipeling

![alt text](img6.png)

## Example Usage

```python
img_path='../img.jpg'
model_op_path='../img.jpg'
gt_path='../img.jpg'

crf = CRF(img_path, model_op_path, gt_path, 30, clr_op=True, pallet=pallet)
CRF_op = crf.start()
plt.imshow(CRF_op)

```
