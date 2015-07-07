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
from scipy.stats import norm
from scipy import ndimage
from matplotlib import pyplot as plt
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def in_idea_blue_rg(I_origin):
    IDEAL_SKY_BGR = (235, 190, 150)
    RANGE = 30
    return (abs(I_origin[...,0]-IDEAL_SKY_BGR[0]) < RANGE) \
        & (abs(I_origin[...,1]-IDEAL_SKY_BGR[1]) < RANGE) \
        & (abs(I_origin[...,2]-IDEAL_SKY_BGR[2]) < RANGE)

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

def _var_lenpos(val, pos):
    '''
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/ndimage/measurements.py#L330
    '''
    return val, len(pos)

def get_sky_ref_patch(f_gray_one_third):
    lbl, nlbl = ndimage.label(f_gray_one_third)
    lbls = np.arange(1, nlbl+1)
    ipdb.set_trace()
    val, lenpos = ndimage.labeled_comprehension(f_gray_one_third, lbl, lbls, _var_lenpos, float, 0, True)
    pos = lenpos.argmax # pos of largest sky ref patch
    mean = np.mean(val[pos])
    variance = np.variance(val[pos])
    return mean, variance

def _get_top_one_third(img):
    return img[0:img.shape[0],...]

def sky_ref_patch_detection(I_origin):
    # blue sky color range:
    # http://colors.findthedata.com/q/402/10857/What-are-the-RGB-values-of-Sky-Blue
    I_gray = cv2.cvtColor(I_origin, cv2.COLOR_BGR2GRAY)
    sky_prob_map = np.zeros(I_gray.shape)
    ipdb.set_trace()
    sky_prob_map[ in_idea_blue_rg(I_origin) ] = 1.0
    _grad_x = np.absolute( cv2.Sobel(I_gray, cv2.CV_64F, 1, 0, ksize=5) )
    _grad_y = np.absolute( cv2.Sobel(I_gray, cv2.CV_64F, 0, 1, ksize=5) )
    _rv = norm(loc=0., scale=255./3.) # 3*sigma = 255, rv ~ random variable
    # ipdb.set_trace()
    sky_prob_map[ (sky_prob_map ==1.0) &
                  ((_grad_x >0.05*255.) | (_grad_y >0.05*255.)) ] \
        = _rv.pdf( (_grad_x + _grad_y)/2.0 )[ (sky_prob_map ==1.0) &
                  ((_grad_x >0.05*255.) | (_grad_y >0.05*255.)) ] # average of gradx, y
    # _mask = np.ones(sky_prob_map.shape, dtype=bool)
    # _mask[sky_prob_map==0.0] = False
    detect_res = cape_util.detect_bimodal( [get_smoothed_hist( cape_util.mask_skin(I_gray, sky_prob_map!=0.0) )] )[0] # detect_bimodal is an array f
    if (detect_res[0] == True): #could return F or None, must be ==True
        sky_prob_map[ I_gray ==detect_res[1] ] = 0.0 #exclude pixels correspond to the dark mode

    S = []
    if ( np.sum( _get_top_one_third(sky_prob_map) ) !=0): # top 1/3 has sky
        ipdb.set_trace()
        for i in range(3): #B, G, R
            masked_I_one_thrid = cape_util.mask_skin( # mask non-blue area 0, copy the rest
                _get_top_one_third(I_origin)[...,i]
                , _get_top_one_third(sky_prob_map)!=0.0 # sky_prob map changed after previous step
            )
            mean, var = get_sky_ref_patch( masked_I_one_thrid )
            S.append(mean)
        _rv_sky_patch = norm(loc=mean, scale=var)
    # ipdb.set_trace()
    return

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    sky_ref_patch_detection(I_origin)
    return 0

if __name__ == '__main__':
    main()
