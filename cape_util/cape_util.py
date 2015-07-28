"""
Created on June 18, 2015
@author: shiruilu

Common utils for CAPE
"""

import cv2
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def safe_convert(x, new_dtype):
    """
    http://stackoverflow.com/a/23325108/2729100
    convert x to new_dtype, clip values larger than max or smaller than min
    """
    info = np.iinfo(new_dtype)
    return x.clip(info.min, info.max).astype(new_dtype)

def detect_bimodal(H):
    """
    H: all the (smoothed) histograms of faces on the image
    RETURN:
    bimodal_Fs: True means detected
                False means undetected (i.e. not bimodal)
                None means not sure, will plot H[i] for analysis
    D, M, B:    *Arrays* of detected Dark, Median, Bright intensities.
                i.e. x-index of H
    """
    # argrelextrema return (array([ 54, 132]),) (a tuple), only [0] used for 1d
    maximas_Fs = [ argrelextrema(h, np.greater, order=10)[0] for h in H ]
    # argrelextrema return (array([ 54, 132]),) (a tuple), only [0] used for 1d
    minimas_Fs = [ argrelextrema(h, np.less, order=10)[0] for h in H ]
    # # to visualize the bimodal:
    # print "maximas each face(hist): ", maximas_Fs \
    #       , "minimas each face(hist): ", minimas_Fs
    # plt.plot(H[i]); plt.xlim([1,256]); plt.show()
    bimodal_Fs = np.zeros(len(H) ,dtype=bool)
    D = np.zeros(len(H)); M = np.zeros(len(H)); B = np.zeros(len(H));
    for i in range(len(H)): # each face i
        tot_face_pix = np.sum(H[i])
        if len(maximas_Fs[i]) ==2 and len(minimas_Fs[i]) ==1: #bimodal detected
            d = maximas_Fs[i][0]
            b = maximas_Fs[i][1]
            m = minimas_Fs[i][0]
            # print 'd,b,m: ',d,b,m
            B[i] = b; M[i] = m; D[i] = d;
            # NOTICE: Here its 0.003 not 5%(as described in CAPE)!
            # 5% should be cumulated from several cylinders around the peak
            # Here it's ONLY the highest peak
            if H[i][d] >=0.003*tot_face_pix and H[i][b] >=0.003*tot_face_pix \
               and (H[i][m] <=0.8*H[i][d] and H[i][m] <=0.8*H[i][b]):
                bimodal_Fs[i] = True
        elif len(maximas_Fs[i]) >2 or len(minimas_Fs[i]) >1:
            print '?? more than two maximas, or more than one minima, see the plot'
            plt.plot(H[i]); plt.xlim([1,256]); plt.show()
            bimodal_Fs[i] = None
        else:
            None
    return bimodal_Fs, D, M, B

def frange(start, stop, step):
    it = start
    while(it < stop):
        yield it
        it += step

def mask_skin(img, mask):
    img_cp = img.copy()
    img_cp[ ~mask ] = 0 # non-skin area set to 0
    return img_cp

def mag(img, dtype='int'):
    """
    magnify from [0,1] to [0.255]
    """
    if dtype == 'int':
        return safe_convert(np.rint(img*255), np.uint8)
    elif dtype == 'float':
        return (img*255)
    elif dtype == 'trim':
        return safe_convert(np.rint(img), np.uint8)
    else:
        raise ValueError('no such data type')

def display(img, name='', mode='bgr'):
    """
    display image using matplotlib

    ARGS:
    img: bgr mode
    name: string, displayed as title
    """
    if mode == 'bgr':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif mode == 'rgb':
        plt.imshow(img)
    elif mode == 'gray':
        plt.imshow(img, 'gray')
    elif mode == 'rainbow':
        plt.imshow(img, cmap='rainbow')
    else:
        raise ValueError('CAPE display: unkown mode')
    plt.title(name)
    plt.show()