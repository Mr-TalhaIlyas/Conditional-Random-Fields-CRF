import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
import tensorflow as tf
from gray2color import gray2color
import imageio
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
'''
This script implements CRF as described in Deeplab paper it takes about 0.2 seconds per image.
'''
class Seg_CRF:
    def __init__(self, img_path, model_op_path, num_of_classes, img_w=1024, img_h=512,
                 apperance_kernel=[8, 164, 100], spatial_kernel=[3, 10], clr_op=False, pallet2use ='ade20k'):
        '''
        Parameters
        ----------
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
        Returns
        -------
        crf op image as array range [0, 1] grayscael or RGB
        '''
        self.img_path = img_path
        self.model_op_path = model_op_path
        self.num_of_classes = num_of_classes
        self.img_w = img_w
        self.img_h = img_h
        self.n_classes = num_of_classes
        self.pallet2use = pallet2use
        self.clr_op = clr_op
        self.apperance_kernel = apperance_kernel
        self.spatial_kernel = spatial_kernel
        # read iamges
        self.image = cv2.imread(self.img_path) 
        self.image = cv2.resize(self.image, (self.img_w, self.img_h))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.mask = cv2.imread(self.model_op_path, 0)
        self.mask = cv2.resize(self.mask, (self.img_w, self.img_h))
        
    def start(self):
        if self.n_classes < 2:
            self.n_classes = 2
        # covnert masks form [W, H] to [W, H , n_classes]
        if self.n_classes <= 2:
            inverse_mask = cv2.bitwise_not(self.mask)
            mask = self.mask[np.newaxis, :, :]
            inverse_mask = inverse_mask[np.newaxis, :, :]
            model_op = np.concatenate([inverse_mask, mask], axis=0)
            
        elif self.n_classes > 2:
            mask = self.mask  #* 255
            mask = tf.cast(mask, 'int32')
            sess1 = tf.compat.v1.Session()
            mask = sess1.run(mask).squeeze()
            mask = tf.one_hot(mask, self.n_classes, axis = -1)
            sess = tf.compat.v1.Session()
            mask = sess.run(mask)
            mask = mask.transpose(2,0,1)
            model_op = mask
        
        feat_first = model_op.reshape((self.n_classes,-1))# Flattening classes (e.g. BG and FG)
        # Uniray Potential
        unary = unary_from_softmax(feat_first)
        unary = np.ascontiguousarray(unary)
        
        d = dcrf.DenseCRF2D(self.image.shape[1], self.image.shape[0], self.n_classes)
        d.setUnaryEnergy(unary)# add uniray potential to paiwise potential
        # Pairwise potential
        # it smooths the masks  # 5     10 sxy=(.5, .5)
        d.addPairwiseGaussian(sxy=self.spatial_kernel[0], compat=self.spatial_kernel[1],
                              kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC) # Spatial/Smootheness Kernel
        
        d.addPairwiseBilateral(sxy=self.apperance_kernel[0], srgb=self.apperance_kernel[1], rgbim=self.image, #5  13 10  sxy=(1.5, 1.5), srgb=(64, 64, 64)
                               compat=self.apperance_kernel[2], kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)  # Appearance/approximity Kernel
        '''
        1. increasing sxy puts more focus on changing intensity
        2. if we increase srgb the pixels in close proximity will be assigned one class 
           (high value will cause the erosion like effect at boundaries)
        '''
        Q = d.inference(1)
        CRF_op = np.argmax(Q, axis=0).reshape((self.image.shape[0], self.image.shape[1])).astype(np.float32) # Becarefull regardnig datatype
       
        if self.clr_op:
            CRF_op_rgb = gray2color(CRF_op, use_pallet=self.pallet2use)    
            return CRF_op.astype(np.uint8), CRF_op_rgb
        else:     
            return CRF_op
#%%        

# Zero pixels are consdered background
# Usage
img_path='D:/Anaconda/Image_analysis/cat.png'
model_op_path='D:/Anaconda/Image_analysis/mask.png'

# Default Values are
apperance_kernel = [8, 164, 100] # PairwiseBilateral [sxy, srgb, compat]  
spatial_kernel = [3, 10]         # PairwiseGaussian  [sxy, compat] 

# or if you want to to specify seprately for each XY direction and RGB color channel then

apperance_kernel = [(1.5, 1.5), (64, 64, 64), 100] # PairwiseBilateral [sxy, srgb, compat]  
spatial_kernel = [(0.5, 0.5), 10]                  # PairwiseGaussian  [sxy, compat] 

crf = Seg_CRF(img_path, model_op_path, 2, img_w=1024, img_h=512,
                 apperance_kernel=apperance_kernel, spatial_kernel=spatial_kernel,
                 clr_op=True, pallet2use ='cityscape')

gray, rgb = crf.start()
plt.imshow(rgb)

















