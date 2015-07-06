"""
Created on July 6, 2015
@author: shiruilu

Sky Enhancement from CAPE, put together
"""
import os
import sys

source_dirs = ['cape_util', 'aindane', 'face_detection'
               , 'skin_detection', 'wls_filter']

for d in source_dirs:
    sys.path.insert( 0, os.path.join(os.getcwd(), '../'+d) )

import cape_util
import aindane
import appendixa_skin_detect as apa_skin
import wls_filter
import docs_face_detector as vj_face

import cv2
import numpy as np
from scipy.stat import norm
from matplotlib import pyplot as plt
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def in_idea_blue_rg(I_origin):
    IDEAL_SKY_BGR = (235, 206, 135)
    RANGE = 30
    return (abs(I_origin[...,0]-IDEAL_SKY_BGR(0)) < RANGE) \
        & (abs(I_origin[...,1]-IDEAL_SKY_BGR(1)) < RANGE) \
        & (abs(I_origin[...,2]-IDEAL_SKY_BGR(2)) < RANGE)

def get_smoothed_hist(I_gray, ksize=30, sigma=10):
    """
    get smoothed hist from a single channel
    +TODO: consider replace the calc of face_enhancement.py _H, H

    ARGs:
    I_gray: MASKED single channle image (not necessarily gray), 0 will not be counted.
    ksize &
    sigma:  For Gaussian kernel, following 3*sigma rule

    RETURN:
    h:      Smoothed hist
    """
    _h = cv2.calcHist([I_gray],[0],None,[255],[1,256]).T.ravel()
    h = np.correlate(_h, cv2.getGaussianKernel(ksize,sigma).ravel(), 'same')
    return h


def sky_ref_patch_detection(I_origin):
    # blue sky color range:
    # http://colors.findthedata.com/q/402/10857/What-are-the-RGB-values-of-Sky-Blue
    sky_prob_map = np.zeros(I_origin.shape)
    sky_prob_map[ in_idea_blue_rg(I_origin) ] = 1.0
    I_gray = cv2.cvtColor(I_origin, cv2.COLOR_BGR2GRAY)
    _grad_x = np.absolute( cv2.Sobel(I_gray, cv2.CV_64F, 1, 0, ksize=5) )
    _grad_y = np.absolute( cv2.Sobel(I_gray, cv2.CV_64F, 0, 1, ksize=5) )
    _rv = norm(loc=0., scale=255./3.) # 3*sigma = 255, rv ~ random variable
    sky_prob_map[ (sky_prob_map ==1.0) &
                  ((_grad_x >0.05*255.) | (_grad_y >0.05*255.)) ] \
        = _rv.pdf( (_grad_x + _grad_y)/2.0 ) # average of gradx, y
    _mask = np.ones(sky_prob_map.shape, dtype=bool)
    _mask[sky_prob_map==0.0] = False
    detect_res = cape_util.detect_bimodal( [get_smoothed_hist( cape_util.mask_skin(I_gray, _mask) )] )[0] # detect_bimodal is an array f
    if (detect_res[0] == True): #could return F or None, must be ==True
        sky_prob_map[ I_gray ==detect_res[1] ] = 0.0 #exclude pixels correspond to the dark mode

    if ( np.sum(sky_prob_map[0:sky_prob_map.shape[0]*1.0/3, ...]) !=0):
        

    return

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    sky_ref_patch_detection(I_origin)
    return 0

if __name__ == '__main__':
    main()