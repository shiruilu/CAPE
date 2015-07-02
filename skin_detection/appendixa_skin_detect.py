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
    neighbor = np.ones([4,4], dtype='float')
    return correlate2d(mask.astype('float')/255.0, neighbor
                       , mode='same', boundary='wrap') >= 1

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
    skinMask[ ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.0, prob=1.0)
              & HSV_threshold(img_HSV[...,0], img_HSV[...,1]) ] \
        = 255
    # print time.time()-s
    # relaxed ellipse test, guarenteed by skin neighborhood
    skinMask[ (skinMask ==0)
              & ellipse_test(img_LAB[...,1], img_LAB[...,2]
                             , bound=1.25, prob=0.9)
              & check_neighbor(skinMask)] = 255
    # print time.time()-s
    # initialization, can't remove, otherwise mask==0 area will be random
    skin = 255*np.ones(img.shape, img.dtype)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    return skin, (skinMask/255).astype(bool)

def main():
    img = cv2.imread(IMG_DIR+'teaser_face.png')
    skin, _ = skin_detect( img )
    plt.imshow( cv2.cvtColor(np.hstack([img, skin]), cv2.COLOR_BGR2RGB) )
    plt.show()
    return 0

if __name__ == '__main__':
    main()