"""
Created on July 22, 2015
@author: shiruilu

Main Function of CAPE, including detail enhacement(last step)
"""
import os
import sys

source_dirs = ['cape_util', 'aindane', 'face_detection'
               , 'skin_detection', 'wls_filter', 'face_skin'
               , 'pySaliencyMap', 'blue_sky', 'shad_saliency_enhance']

for d in source_dirs:
    sys.path.insert( 0, os.path.join(os.getcwd(), d) )

import cape_util
import appendixa_skin_detect as apa_skin
import wls_filter
import face_enhancement
import sky_enhancement
import ss_enhance

import cv2
import numpy as np
from matplotlib import pyplot as plt
import ipdb

IMG_DIR = 'resources/images/'

# # tricky way of removing ipdb breakpoints
# def f(): pass
# ipdb.set_trace = f

def detail_enhace(I, skin_prob_map, sky_prob_map, c=0.2):
    """
    Detail Enhancement as described in Section 7
    @I : uint8, In CIELab format
    @skin_prob_map : float, 0-1
    @sky_prob_map : float, 0-1
    """
    # skin_prob_map and sky_prob_map are 0-1
    assert 0<=c<=0.25
    P_ns = (skin_prob_map+sky_prob_map)/(skin_prob_map+sky_prob_map).max()
    I_lab = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    L = I_lab[...,0]
    _,Detail = wls_filter.wlsfilter(L)
    Detail = Detail*255
    L_new = L + c*P_ns*Detail
    I_lab[...,0] = cape_util.safe_convert(L_new, np.uint8)
    return cv2.cvtColor(I_lab, cv2.COLOR_LAB2BGR)

def main():
    I_org = cv2.imread(IMG_DIR+'input_teaser.png')
    skin_prob_map = apa_skin.skin_prob_map(I_org)
    lambda_ = 1.0
    res_skin = face_enhancement.face_enhancement(I_org, lambda_)
    res_sky, sky_prob_map = sky_enhancement.sky_enhancement(res_skin)
    res_ss = ss_enhance.ss_enhance(res_sky)
    res_de = detail_enhace(res_ss, skin_prob_map, sky_prob_map)
    # res_de = res_ss
    cape_util.display( np.hstack([I_org,res_de]), name='lambda_='+str(lambda_) )
    return 0

if __name__ == '__main__':
    main()
