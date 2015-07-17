"""
Created on July 16, 2015
@author: shiruilu

Shadowed Saliency Enhancement
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

from math import sqrt
import cv2
import numpy as np
from scipy.stats import norm
from scipy import ndimage as ndi
from scipy import optimize as opt
from matplotlib import pyplot as plt
import ipdb

# tricky way of removing ipdb breakpoints
def f(): pass
ipdb.set_trace = f

def G(arr_2d, sigma=5):
    _gaus_x = cv2.getGaussianKernel(3*sigma, sigma)
    gaus = _gaus_x * _gaus_x.T
    return ndi.convolve(arr_2d, gaus) # default mode is 'reflect'

def abs_grad(arr_2d):
    dx, dy = np.gradient(arr_2d)
    return np.absolute(dx) + np.absolute(dy)

def get_w_spacial(n,m):
    """
    ARGs:
    n,m: array's shape (n,m)
    """
    xc, yc = n/2, m/2
    maxdE = math.sqrt( (n/2)**2 + (m/2)**2 )

    # construct an array, each element (x,y) is it's pos
    xy = np.array.empty(n, m, 2)
    for i in range(n):
        xy[i,:,0] = i
    for j in range(m):
        xy[:,j:1] = j

    return 1 - ( np.sqrt((xy[...,0]-xc)**2 + (xy[...,1]-yc)**2) / maxdE )**2

def get_energy_map(I, skin_mask):
    """
    ARGs:
    I: in CIELab format, 3 channels
    skin_mask: 2d, dtype=bool. True for skin, False for not
    """
    I_gray = cv2.cvtColor(I, cv2.COLOR_LAB2GRAY)
    _e_fine = abs_grad(I_gray)
    _sigma = 5
    _e_mul = _e_fine + _sigma*abs_grad( G(I_gray, _sigma) )
    _e_color = _sigma * ( (abs_grad(G(I[:,:,1]), _sigma))   # CIE a
                         + abs_grad(G(I[:,:,2]), _sigma) )  # CIE b
    e_grad = _e_mul + _e_color

    e_H =
    e_face =
    w_spacial = get_w_spacial(I_gray.shape)

    e_base = (e_grad + e_H + e_face) * w_spacial
    return e_base + e_grad # e_improved

def main():
    return

if __name__ == '__main__':
    main()