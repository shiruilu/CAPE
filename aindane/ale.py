"""
Created on June 9, 2015
@author: shiruilu

Adaptive Luminance Enhancement from AINDANE
"""

import numpy as np
import cv2
import time

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def ale(img_path):
    """algorithm"""
    I = cv2.imread(img_path)
    #?cv2 doing same as NTSC I_NTSC = (76.245*I[..., 2] + 149.685*I[..., 1] + 29.07*I[..., 0])/255.0 # equation 1
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    In = I/255.0 #2d array, equation 2
    #hist, bins = np.histogram(I.flatten(), bins=256, range=[0,256])
    hist = cv2.calcHist([I],[0],None,[256],[0,256])
    cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max()/ cdf.max()
    L = np.searchsorted(cdf, 0.1*I.shape[0]*I.shape[1], side='right') # http://stackoverflow.com/a/25032853/2729100
    La = np.array([L]) # L as array
    z = np.piecewise(La, [La<=50, La>50 and La<=150, La>150], [0, (L-50)/100.0, 1]) #http://stackoverflow.com/a/16575522/2729100

    In_prime = (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z)) /2.0
    return 0


def test_ale():
    ale(IMG_DIR+'pdbuse.png')
    return 0

if __name__ == '__main__':
    test_ale()