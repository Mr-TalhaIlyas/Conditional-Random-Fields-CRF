import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
import tensorflow as tf
import imageio
'''
This script implements CRF as described in Deeplab paper it takes about 0.2 seconds per image
it takes three inputs.
1.
2.
3.
'''
n_classes = 30
i = 7
img_w = 1024
img_h = 512

image = cv2.imread('C:/Users/Talha/Desktop/img_2032.jpg') #/ 255    #D:/Anaconda/Image_analysis/cat.png
image = cv2.resize(image, (img_w, img_h))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_o = cv2.imread('C:/Users/Talha/Desktop/img_2032_m.jpg',0)# / 255
mask = cv2.resize(mask_o, (img_w, img_h))
mask = mask_o
GT = cv2.imread('C:/Users/Talha/Desktop/img_2032_m.jpg',0)

if n_classes <= 2:
    inverse_mask = cv2.bitwise_not(mask)
    mask = mask[np.newaxis, :, :]
    inverse_mask = inverse_mask[np.newaxis, :, :]
    model_op = np.concatenate([inverse_mask, mask], axis=0)
    
elif n_classes > 2:
    mask = mask  #* 255
    mask = tf.cast(mask, 'int32')
    sess1 = tf.compat.v1.Session()
    mask = sess1.run(mask).squeeze()
    mask = tf.one_hot(mask, n_classes, axis = -1)
    sess = tf.compat.v1.Session()
    mask = sess.run(mask)
    mask = mask.transpose(2,0,1)
    model_op = mask



feat_first = model_op.reshape((n_classes,-1))# Flattening classes (e.g. BG and FG)
# Uniray Potential
unary = unary_from_softmax(feat_first)
unary = np.ascontiguousarray(unary)

d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_classes)
d.setUnaryEnergy(unary)# add uniray potential to paiwise potential
# Pairwise potential
# it smooths the masks  # 5     10 sxy=(.5, .5)
d.addPairwiseGaussian(sxy=(3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC) # Spatial/Smootheness Kernel
d.addPairwiseBilateral(sxy=(8), srgb=(164), rgbim=image, #5  13 10  sxy=(1.5, 1.5), srgb=(64, 64, 64)
                       compat=100,
                       kernel=dcrf.DIAG_KERNEL,
                       normalization=dcrf.NORMALIZE_SYMMETRIC)  # Appearance/approximity Kernel
'''
1. increasing sxy puts more focus on changing intensity
2. if we increase srgb the pixels in close proximity will be assigned one class 
   (high value will cause the erosion like effect at boundaries)
'''
Q = d.inference(1)
CRF_op = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1])).astype(np.float32) # Becarefull regardnig datatype

crf_rgb = cv2.cvtColor(CRF_op, cv2.COLOR_GRAY2RGB)*255
overlayed_result = cv2.addWeighted(crf_rgb.astype(np.uint8), 0.5, image, 0.7, 0)

fig, axs = plt.subplots(2, 2, figsize = (5, 5))
#fig.suptitle('From Topto Bottom Original_Img, Ground_Truth, Predicted')

axs[0,0].imshow(image, interpolation = 'bilinear')
axs[0,0].axis("off")
axs[0,0].set_title('Original_Img')
axs[0,1].imshow(cv2.resize(mask_o, (img_w, img_h)), cmap = 'gray', interpolation = 'bilinear')
axs[0,1].axis("off")
axs[0,1].set_title('Model ouput')
axs[1,0].imshow(CRF_op, cmap = 'gray', interpolation = 'bilinear')
axs[1,0].axis("off")
axs[1,0].set_title('CRF ouput')
axs[1,1].imshow(overlayed_result,  interpolation = 'bilinear')
axs[1,1].axis("off")
axs[1,1].set_title('Overlayed_Result')
#imageio.imwrite('C:/Users/Talha/Desktop/overlayed_result_202.png', overlayed_result)
#%%
##################################################################################################
# For coloring the grayscale op
##################################################################################################
def gray2rgb(gray_processed, pallet):
    
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

pallet_28 =  np.array([[[ 0,  0,  0],
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




gray = tf.cast(CRF_op, 'int32')
sess1 = tf.compat.v1.Session()
gray = sess1.run(gray)
encoded_op = tf.one_hot(gray, n_classes, axis = -1)
sess = tf.compat.v1.Session()
encoded_op = sess.run(encoded_op)
CRF_op_clr = np.argmax(encoded_op, 2)


gray = mask_o #* (n_classes+1)
gray = tf.cast(gray, 'int32')
sess1 = tf.compat.v1.Session()
gray = sess1.run(gray)
encoded_op = tf.one_hot(gray, n_classes, axis = -1)
sess = tf.compat.v1.Session()
encoded_op = sess.run(encoded_op)
mask_o_clr = np.argmax(encoded_op, 2)

gray = tf.cast(GT, 'int32')
sess1 = tf.compat.v1.Session()
gray = sess1.run(gray)
encoded_op = tf.one_hot(gray, n_classes, axis = -1)
sess = tf.compat.v1.Session()
encoded_op = sess.run(encoded_op)
GT_clr = np.argmax(encoded_op, 2)

CRF_op_clr = gray2rgb(CRF_op_clr, pallet_28)
mask_o_clr = gray2rgb(mask_o_clr, pallet_28)
GT_clr = gray2rgb(GT_clr, pallet_28)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

ax1.imshow(mask_o_clr, cmap = 'gray', interpolation = 'bilinear')
ax1.axis("off")
ax1.set_title('Before CRF')

ax2.imshow(CRF_op_clr, interpolation = 'bilinear')
ax2.axis("off")
ax2.set_title('After CRF')
#%%
imageio.imwrite('C:/Users/Talha/Desktop/results/CRF_op_clr_{}.png'.format(i), (CRF_op_clr * 255))
imageio.imwrite('C:/Users/Talha/Desktop/results/mask_o_clr_{}.png'.format(i), (mask_o_clr * 255))
imageio.imwrite('C:/Users/Talha/Desktop/results/GT_clr_{}.png'.format(i), (GT_clr * 255))
imageio.imwrite('C:/Users/Talha/Desktop/results/overlayed_result_{}.png'.format(i), overlayed_result)






















