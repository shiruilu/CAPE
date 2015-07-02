"""
Created on May 27, 2015
@author: shiruilu

skin detect from Appendix A of CAPE
"""

import numpy as np
import cv2
# import time
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def ellipse_test(A, B, bound=1.0, prob=1.0):
    '''test a CIELab color falls in certain ellipse'''
    return (1.0*(A-143)/6.5)**2 + (1.0*(B-148)/12.0)**2 < (1.0*bound/prob)

def check_neighbor(mask):
    '''for relaxed 4-neighbor test'''
    """
    width, height = mask.shape
    sum = 0
    for x in range(-12,13,2):
        for y in range(-12,13,2):
            if mask[(i+x)%width, (j+y)%height] == 255:
                sum = sum + 1
    return sum >= 1
    """
    neighbor = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype='float')
    return correlate2d(mask.astype('float')/255.0, neighbor, mode='same', boundary='wrap') >= 1

def HSV_threshold(H, S):
    #s:0.25~0.75, h>0.095
    return (S>=0.25*255) & (S<=0.75*255) & (H<0.095*180)

def skin_detect(img):
    """img: in BGR mode"""
    # initialized all-white mask
    skinMask = np.zeros(img.shape[0:2], img.dtype)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float')
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')

    # s = time.time()
    # ellipse and HSV_test
    '''
    for (i,j), value in np.ndenumerate(skinMask):
        if ellipse_test(img_LAB[i,j][1], img_LAB[i,j][2], bound=1.0, prob=1.0):
            skinMask[i,j] = 255
               # and HSV_threshold(img_HSV[i,j][0], img_HSV[i,j][1]) \
    '''
    # ipdb.set_trace()
    skinMask[ ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.0, prob=1.0)
              & HSV_threshold(img_HSV[...,0], img_HSV[...,1]) ] \
        = 255
    
    # print time.time()-s
    # relaxed ellipse test, guarenteed by skin neighborhood
    """
    for (i,j), value in np.ndenumerate(skinMask):
        if skinMask[i,j] ==0:
            skinMask[i,j] = 255 \
                if ellipse_test(img_LAB[i,j][1], img_LAB[i,j][2], \
                                bound=1.25, prob=0.9) \
                   and check_neighbor(skinMask, i, j) \
                else 0
    """
    mask_cp = skinMask.copy()
    # ipdb.set_trace()
    skinMask[ (mask_cp ==0) & ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.25, prob=0.9) & check_neighbor(mask_cp)] = 255
    # skinMask[ ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.25, prob=0.9) ] = 255
    # print time.time()-s
    # initialization, can't remove, otherwise mask==0 area will be random
    skin = 255*np.ones(img.shape, img.dtype)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    # if ( not (skin.shape[:2] == skinMask.shape).all() ):
        # print 'weired: ', skin.shape[:2], skinMask.shape
    return skin, (skinMask/255).astype(bool)

def main():
    img = cv2.imread(IMG_DIR+'teaser_face.png')
    skin, _ = skin_detect( img )
    plt.imshow( cv2.cvtColor(np.hstack([img, skin]), cv2.COLOR_BGR2RGB) )
    plt.show()
    return 0

if __name__ == '__main__':
    main()