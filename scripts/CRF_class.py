import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
import tensorflow as tf
import imageio
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
'''
This script implements CRF as described in Deeplab paper it takes about 0.2 seconds per image.
'''
n_classes = 30
pallet =  np.array([[[ 0,  0,  0],
                    [ 0,  0,  0],
                    [ 0,  0,  0],
                    [ 0,  0,  0],
                    [ 0,  0,  0],
                    [111, 74,  0],
                    [ 81,  0, 81],
                    [128, 64,128],
                    [244, 35,232],
                    [250,170,160],
                    [230,150,1400],
                    [ 70, 70, 70],
                    [102,102,156],
                    [190,153,153],
                    [180,165,180],
                    [150,100,100],
                    [150,120, 90],
                    [153,153,153],
                    [153,153,153],
                    [250,170, 30],
                    [220,220,  0],
                    [107,142, 35],
                    [152,251,152],
                    [70,130,180],
                    [220, 20, 60],
                    [255,  0,  0],
                    [ 0,  0,142],
                    [ 0,  0, 70],
                    [ 0, 60,100],
                    [ 0,  0, 90],
                    [0,  0,110],
                    [ 0, 80,100],
                    [ 0,  0,230],
                    [119, 11, 32]]], np.uint8) / 255
class Seg_CRF:
    '''

    Parameters
    ----------
    img_path : path to an image, 
                    Format [H, W, 3]; values ranging from [0, 255]
    model_op_path : path model output of the same input image.
                    Format [H, W]; values ranging from [0, num_of_classes]
    num_of_classes : number of classes in a dataset e.g. in cityscape has 30 classes
    clr_op : color the output or not a bool
    pallet: a [1 x no of classes x 3] array containing the RGB values of classes, type float32
    
    Returns
    -------
    crf op image as array range [0, 1] grayscael or RGB
    
    '''
    def __init__(self, img_path, model_op_path, num_of_classes, clr_op=False, pallet = None):
        
        self.img_path = img_path
        self.model_op_path = model_op_path
        self.num_of_classes = num_of_classes
        self.img_w = 1024
        self.img_h = 512
        self.n_classes = num_of_classes
        self.pallet = pallet
        self.clr_op = clr_op
        # read iamges
        self.image = cv2.imread(self.img_path) #/ 255    #D:/Anaconda/Image_analysis/cat.png
        self.image = cv2.resize(self.image, (self.img_w, self.img_h))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.mask = cv2.imread(self.model_op_path, 0)# / 255
        self.mask = cv2.resize(self.mask, (self.img_w, self.img_h))
        
    def start(self):
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
        d.addPairwiseGaussian(sxy=(3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC) # Spatial/Smootheness Kernel
        d.addPairwiseBilateral(sxy=(8), srgb=(164), rgbim=self.image, #5  13 10  sxy=(1.5, 1.5), srgb=(64, 64, 64)
                               compat=100,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)  # Appearance/approximity Kernel
        '''
        1. increasing sxy puts more focus on changing intensity
        2. if we increase srgb the pixels in close proximity will be assigned one class 
           (high value will cause the erosion like effect at boundaries)
        '''
        Q = d.inference(1)
        CRF_op = np.argmax(Q, axis=0).reshape((self.image.shape[0], self.image.shape[1])).astype(np.float32) # Becarefull regardnig datatype
        if self.clr_op:
           CRF_op = self.gray2rgb(CRF_op, self.pallet)
        return CRF_op
    
    def gray2rgb(self, gray_processed, pallet):

        gray = gray_processed
        w, h = gray.shape
        gray = gray[:,:,np.newaxis]
        gray = tf.image.grayscale_to_rgb((tf.convert_to_tensor(gray)))
        sess = tf.compat.v1.Session()
        gray = sess.run(gray)
        gray = tf.cast(gray, 'int32')
        sess1 = tf.compat.v1.Session()
        gray = sess1.run(gray)
        unq = np.unique(gray)
        rgb = np.zeros((w,h,3))
        
        for i in range(len(unq)):
            clr = pallet[:,unq[i],:]
            rgb = np.where(gray!=unq[i], rgb, np.add(rgb,clr))
        return rgb
#%%        
# Usage
img_path='D:/Anaconda/Datasets/vistas/vistas/val/images/images/_1Gn_xkw7sa_i9GU4mkxxQ.jpg'
model_op_path='D:/Anaconda/Datasets/vistas/vistas/val/masks/masks/_1Gn_xkw7sa_i9GU4mkxxQ.png'

crf = Seg_CRF(img_path, model_op_path, 66, clr_op=True, pallet=pallet)
CRF_op = crf.start()
plt.imshow(CRF_op)

















