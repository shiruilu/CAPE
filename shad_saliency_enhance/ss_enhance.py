"""
Created on July 16, 2015
@author: shiruilu

Shadowed Saliency Enhancement
"""
import os
import sys

source_dirs = ['cape_util', 'aindane', 'face_detection'
               , 'skin_detection', 'wls_filter', 'face_skin']

for d in source_dirs:
    sys.path.insert( 0, os.path.join(os.getcwd(), '../'+d) )

import cape_util
import aindane
import appendixa_skin_detect as apa_skin
import wls_filter
import docs_face_detector as vj_face
import eacp

from math import sqrt
import cv2
import numpy as np
from scipy.stats import norm
from scipy import ndimage as ndi
from scipy import optimize as opt
from matplotlib import pyplot as plt
import ipdb

IMG_DIR = '../resources/images/'

# # tricky way of removing ipdb breakpoints
# def f(): pass
# ipdb.set_trace = f

def G(arr_2d, sigma):
    _gaus_x = cv2.getGaussianKernel(3*sigma, sigma)
    gaus = _gaus_x * _gaus_x.T
    return ndi.convolve(arr_2d, gaus) # default mode is 'reflect'

def abs_grad(arr_2d):
    dx, dy = np.gradient(arr_2d)
    return np.absolute(dx) + np.absolute(dy)

def get_w_spacial((n,m)):
    """
    ARGs:
    n,m: array's shape (n,m)
    """
    xc, yc = n/2, m/2
    maxdE = sqrt( (n/2)**2 + (m/2)**2 )

    # construct an array, each element (x,y) is it's pos
    xy = np.empty([n, m, 2])
    for i in range(n):
        xy[i,:,0] = i
    for j in range(m):
        xy[:,j:1] = j

    return 1 - ( np.sqrt((xy[...,0]-xc)**2 + (xy[...,1]-yc)**2) / maxdE )**2

def get_eH(I):
    """
    need numba jit

    ARGs:
    I: in CIELab format, 3 channels
    """
    eps = 1e-5
    Hab,xedges,yedges = np.histogram2d(I[...,1].ravel(), I[...,2].ravel(), bins=10, normed=True)
    xinterval = xedges[1] - xedges[0]; xmin = xedges[0]
    yinterval = yedges[1] - yedges[0]; ymin = yedges[0]
    e_H = np.empty_like(I[...,1], dtype='float')
    for (x,y),a in np.ndenumerate(I[...,1]):
        b = I[x,y,2]
        H = Hab[int((a-xmin)/xinterval-eps), int((b-ymin)/yinterval-eps)]
        if H >0.015:
            e_H[x,y] = 1.0/(H*100)
        else:
            e_H[x,y] = 1.0/(0.015*100)

    return e_H

def get_energy_map(I, skin_mask, _thresh=0.4):
    """
    ARGs:
    I: in CIELab format, 3 channels
    skin_mask: 2d, dtype=bool. True for skin, False for not
    _thresh: threshold energy for the skin area
    RETURN:
    energy_map: 0-1
    """
    I_gray = cv2.cvtColor(cv2.cvtColor(I, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
    _e_fine = abs_grad(I_gray)
    _sigma = 15
    _e_fine_improve = _sigma*abs_grad( G(I_gray, _sigma) )
    _e_mul = _e_fine + _e_fine_improve
    _e_color = _sigma * ( abs_grad(G(I[:,:,1], _sigma))   # CIE a
                         + abs_grad(G(I[:,:,2], _sigma)) )  # CIE b
    e_grad = _e_mul + _e_color
    e_grad /= e_grad.max() # rescale to 0-1

    e_H = get_eH(I)
    e_face = 1.0 *_thresh *skin_mask
    w_spacial = get_w_spacial(I_gray.shape)

    # ipdb.set_trace()
    e_base = (e_grad + e_H + e_face) * w_spacial
    e_base = eacp.EACP(e_base, I[...,0])
    return e_base/e_base.max()

def ss_enhace(energy_map, L, I_bgr):
    """
    ARGs:
    energy_map: 
    L: opencv CIELab luminance channel (0-255)
    I_bgr: original image in BGR mode
    """
    # cape_util.display(energy_map)
    plt.imshow(energy_map, cmap='rainbow'); plt.show()
    # using reduce: http://stackoverflow.com/a/21817093/2729100
    dark_mask = (L<50) & ( np.maximum.reduce(I_bgr[...,0], I_bgr[...,1], I_bgr[...,2])
                             - np.minimum.reduce(I_bgr[...,0], I_bgr[...,1], I_bgr[...,2])>5 ) #exclude from dark whose max and min of BGR exceeds 5
    Dark = L[dark_mask]
    Bright = L[~dark_mask]
    Dark_Smoothed, _ = wls_filter.wlsfilter(Dark)
    f_sal = min( 2.0, 1.0*np.percentile(Bright, 35)/np.percentile(Dark_Smoothed, 95))

    B, Detail = wls_filter.wlsfilter(L)
    B_new = f_sal * energy_map*B + (1-energy_map)*B
    return B+Detail

def get_skin_mask(I_lab, I_bgr):
    I_aindane = aindane.aindane(I_bgr)
    # cape_util.display(np.hstack([I_bgr,I_aindane])) #debug
    faces_xywh = vj_face.face_detect(I_aindane)
    # faces_xywh = vj_face.face_detect(I_bgr)
    faces = [ I_bgr[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ]
    # for fc in faces: # debug
        # cape_util.display(fc)
    skin_mask = np.zeros(I_bgr.shape[0:2] ,dtype=bool)
    for i, face in enumerate(faces):
        (x,y,w,h) = faces_xywh[i]
        # skin_mask[y:y+h, x:x+w] |= (apa_skin.skin_detect(face)[1]) # only use skinMask
        skin_mask[y:y+h, x:x+w] = True

    return skin_mask

def _test():
    I_bgr = cv2.imread(IMG_DIR+'ss_test_tower.png')
    I_lab = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2LAB)
    skin_mask = get_skin_mask(I_lab, I_bgr)
    energy_map = get_energy_map(I_lab, skin_mask)
    L_new = ss_enhace(energy_map, I_lab[...,0], I_bgr)
    I_lab[...,0] = L_new
    I_res_bgr = cv2.cvtColor(I_lab, cv2.COLOR_LAB2BGR)
    cape_util.display(np.hstack([I_bgr, I_res_bgr]))
    return

if __name__ == '__main__':
    _test()