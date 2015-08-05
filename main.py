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
import ipdb

IMG_DIR = 'resources/images/'

# # tricky way of removing ipdb breakpoints
# def f(): pass
# ipdb.set_trace = f

def global_enhace(I_bgr, skin_prob_map, sky_prob_map):
    """
    Keyword Arguments:
    I_bgr         -- uint8, In BGR format
    skin_prob_map -- float, 0-1
    sky_prob_map  -- float, 0-1
    """
    skin_mask = skin_prob_map > 0.95
    sky_mask  = sky_prob_map  > 1000.0
    # stretch the image contrast to full range by clipping 0.5% of the d/b
    _I_bgr_f = I_bgr.astype('float')
    I_intensity = (_I_bgr_f[...,0] + _I_bgr_f[...,1] + _I_bgr_f[...,2])/3.0
    low  = np.percentile(I_intensity[~skin_mask & ~sky_mask], 0.5 )
    high = np.percentile(I_intensity[~skin_mask & ~sky_mask], 99.5)
    out_low = 1.0; out_high = 255.0; # leave out 0 for masking (skin/sky)
    ratio = 1.0*(out_high - out_low) / (high - low)
    I_bgr_stretched = I_bgr.copy()
    for i in range(3):
        I_bgr_stretched[...,i][~skin_mask & ~sky_mask]\
          = cape_util.mag(
                (_I_bgr_f[...,i] - low) * ratio + out_low, 'trim'
            )[~skin_mask & ~sky_mask]

    # increase the saturation of each pixel by 20%
    # 180*1.2 < 255, no need to worry about overflow
    I_hsv_s = cv2.cvtColor(I_bgr_stretched, cv2.COLOR_BGR2HSV)
    I_hsv_s[...,1][~skin_mask & ~sky_mask] = np.minimum(I_hsv_s[...,1][~skin_mask & ~sky_mask]*1.2, 180)

    return cv2.cvtColor(I_hsv_s, cv2.COLOR_HSV2BGR)

def detail_enhace(I, skin_prob_map, sky_prob_map, c=0.2):
    """
    Detail Enhancement as described in Section 7
    I             -- uint8, In CIELab format
    skin_prob_map -- float, 0-1
    sky_prob_map  -- float, 0-1
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
    img_name = 'pic23.jpg'
    I_org = cv2.imread(IMG_DIR+ img_name)
    skin_prob_map = apa_skin.skin_prob_map(I_org)
    lambda_ = 120
    res_skin = I_org
    res_skin = face_enhancement.face_enhancement(I_org, lambda_)
    cape_util.display( np.hstack([I_org,res_skin]), name='res_skin' )
    res_sky, sky_prob_map = sky_enhancement.sky_enhancement(res_skin)
    cape_util.display( np.hstack([I_org,res_sky]), name='res_sky' )
    # res_ge = global_enhace(res_sky, skin_prob_map ,sky_prob_map)
    res_ge = global_enhace(res_skin, skin_prob_map ,sky_prob_map)
    cape_util.display( np.hstack([I_org,res_ge]), name='res_ge' )
    res_ss = ss_enhance.ss_enhance(res_ge)
    cape_util.display( np.hstack([I_org,res_ss]), name='res_ss' )
    res_de = detail_enhace(res_ss, skin_prob_map, sky_prob_map)
    cape_util.display( np.hstack([I_org,res_de]), name='res_de' )
    # res_de = res_ss
    cape_util.display( np.hstack([I_org,res_de]), name='lambda_='+str(lambda_) )
    DIR = './resources/results/'
    cv2.imwrite( DIR+str.split(img_name, '.')[0]+'_res.png', np.hstack([I_org, res_de]) )
    return 0

if __name__ == '__main__':
    main()
