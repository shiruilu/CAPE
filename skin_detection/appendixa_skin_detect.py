"""
Created on May 27, 2015
@author: shiruilu

skin detect from Appendix A of CAPE
"""
import os
import sys

source_dirs = ['cape_util']

for d in source_dirs:
    sys.path.insert( 0, os.path.join(os.getcwd(), d) )

import cape_util

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def ellipse_test(A, B, bound=1.0, prob=1.0, return_prob=False):
    '''test a CIELab color falls in certain ellipse'''
    elpse = (1.0*(A-143)/6.5)**2 + (1.0*(B-148)/12.0)**2
    if not return_prob:
        return elpse < (1.0*bound/prob)
    else:
        return np.minimum(1.0/(elpse+1e-6), 1.0)

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

    # ellipse and HSV_test
    skinMask[ ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.0, prob=1.0)
              & HSV_threshold(img_HSV[...,0], img_HSV[...,1]) ] \
        = 255
    # relaxed ellipse test, guarenteed by skin neighborhood
    skinMask[ (skinMask ==0)
              & ellipse_test(img_LAB[...,1], img_LAB[...,2]
                             , bound=1.25, prob=0.9)
              & check_neighbor(skinMask)] = 255
    # filling holes:image closing on skinMask
    # http://stackoverflow.com/a/10317883/2729100
    _h,_w = img.shape[0:2]
    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(_h/10,_w/10))
    # skinMask_closed = cv2.morphologyEx(skinMask,cv2.MORPH_CLOSE,_kernel)
    skinMask_closed = skinMask
    cape_util.display(np.hstack([skinMask, skinMask_closed]), name='skin mask closing before after', mode='gray')
    # initialization, can't remove, otherwise mask==0 area will be random
    skin = 255*np.ones(img.shape, img.dtype)
    skin = cv2.bitwise_and(img, img, mask=skinMask_closed)
    return skin, (skinMask_closed/255).astype(bool)

def skin_prob_map(img):
    """
    Keyword Arguments:
    img -- np.uint8 (m,n,3) BGR
    """
    # initialized all-white mask
    skin_prob_map = np.zeros(img.shape[0:2], dtype='float')
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float')
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float')
    skin_prob_map = ellipse_test(img_LAB[...,1], img_LAB[...,2], bound=1.0, prob=1.0, return_prob=True)
    skin_prob_map[HSV_threshold(img_HSV[...,0], img_HSV[...,1]) ] = 0.0
    skin_prob_map[ (skin_prob_map < 1.0)
              & ellipse_test(img_LAB[...,1], img_LAB[...,2]
                             , bound=1.25, prob=0.9)
                   & check_neighbor(255*(skin_prob_map==1.0).astype('uint8'))] = 1.0
    return skin_prob_map

def main():
    img = cv2.imread(IMG_DIR+'teaser_face.png')
    skin, _ = skin_detect( img )
    plt.imshow( cv2.cvtColor(np.hstack([img, skin]), cv2.COLOR_BGR2RGB) )
    plt.show()
    return 0

if __name__ == '__main__':
    main()