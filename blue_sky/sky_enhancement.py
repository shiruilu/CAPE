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
from scipy import ndimage as ndi
from scipy import optimize as opt
from matplotlib import pyplot as plt
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def in_idea_blue_rg(I_origin):
    img = I_origin.astype(float)
    IDEAL_SKY_BGR = (225, 190, 170)
    RANGE = (25, 30, 50)
    return (abs(img[...,0]-IDEAL_SKY_BGR[0]) < RANGE[0]) \
        & (abs(img[...,1]-IDEAL_SKY_BGR[1]) < RANGE[1]) \
        & (abs(img[...,2]-IDEAL_SKY_BGR[2]) < RANGE[2])

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
    lbl, nlbl = ndi.label(f_gray_one_third)
    lbls = np.arange(1, nlbl+1)
    # ipdb.set_trace()
    _res = ndi.labeled_comprehension(f_gray_one_third, lbl, lbls, _var_lenpos, object, 0, True)
    val = []; lenpos = [];

    for idx, (_val, _lenpos) in np.ndenumerate(_res):
        val.append(_val); lenpos.append(_lenpos)

    pos = np.array(lenpos).argmax() # pos of largest sky ref patch
    print 'max patch amount, patch: ', lenpos[pos], val[pos]
    mean = np.mean(val[pos])
    variance = np.var(val[pos])
    return mean, variance

def _get_top_one_third(img):
    return img[0:img.shape[0]*1.0/3,...]


def sky_prob_to_01(sky_prob_map, thresh=0.0):
    return sky_prob_map>thresh

def sky_ref_patch_detection(I_origin):
    """
    RETURN:
    S: list:[Sb, Sg, Sr]
    sky_prob_map
    """
    I_gray = cv2.cvtColor(I_origin, cv2.COLOR_BGR2GRAY)
    sky_prob_map = np.zeros(I_gray.shape)
    sky_prob_map[ in_idea_blue_rg(I_origin) ] = 1.0

    _grad_x = np.absolute( ndi.prewitt(I_gray, axis=1 ,mode='nearest') )
    _grad_y = np.absolute( ndi.prewitt(I_gray, axis=0 ,mode='nearest') )
    _rv = norm(loc=0., scale=255./3.) # 3*sigma = 255, rv ~ random variable

    _grad_percent = 0.10 # 0.05 of the original paper
    sky_prob_map[ (sky_prob_map ==1.0) &
                  ((_grad_x >_grad_percent*255.) | (_grad_y >_grad_percent*255.)) ] \
        = _rv.pdf( (_grad_x + _grad_y)/2.0 )[ (sky_prob_map ==1.0) &
                                              ((_grad_x >_grad_percent*255.) | (_grad_y >_grad_percent*255.)) ] # average of gradx, y


    _L = cv2.cvtColor(I_origin, cv2.COLOR_BGR2LAB)[...,0]
    detect_res = cape_util.detect_bimodal( [get_smoothed_hist( cape_util.mask_skin(_L, sky_prob_map!=0.0) )] )[0] # detect_bimodal is an array f
    if (detect_res[0] == True): #could return F or None, must be ==True
        print 'bimodal detected in current sky_ref_patch'
        sky_prob_map[ I_gray ==detect_res[1] ] = 0.0 #exclude pixels correspond to the dark mode

    S = []
    if ( np.sum( _get_top_one_third(sky_prob_map) ) !=0): # top 1/3 has sky
        for i in range(3): #B, G, R
            masked_I_one_thrid = cape_util.mask_skin( # mask non-blue area 0, copy the rest
                _get_top_one_third(I_origin)[...,i]
                , _get_top_one_third(sky_prob_map)!=0.0 # sky_prob map changed after previous step
            )
            mean, var = get_sky_ref_patch( masked_I_one_thrid )
            S.append(mean)
            _rv_sky_patch = norm(loc=mean, scale=var)

    print 'S(b,g,r): ',S
    return S, sky_prob_map

def sky_cloud_decompose(Sbgr, sky_prob_map ,sky_pixels, lambda_=0.2):

    only_sky_pixels = sky_pixels[ sky_prob_to_01(sky_pixels[...,0]) ]

    def J_prime(x):
        Alpha = x[0]; C = x[1]; S=x[2]
        def u(i):
            return x[0]*x[1] + (1-x[0])*x[2]*Sbgr[i] - only_sky_pixels[...,i]
        J_alpha = 2*( u(0)*(C-S*Sbgr[0]) + u(1)*(C-S*Sbgr[1]) + u(2)*(C-S*Sbgr[2]) )
        J_c = (2*(u(0)+u(1)+u(2)) * 3*Alpha)
        J_s = ( (2*(1-Alpha)*(u(0)*Sbgr[0]+u(1)*Sbgr[1]+u(2)*Sbgr[2])) + 2*lambda_*(S-1) )

        return np.array( (J_alpha, J_c, J_s) )

    def J(x):
        return (
            (x[0]*x[1] + (1-x[0])*x[2]*Sbgr[0] - only_sky_pixels[...,0])**2
            + (x[0]*x[1] + (1-x[0])*x[2]*Sbgr[1] - only_sky_pixels[...,1])**2
            + (x[0]*x[1] + (1-x[0])*x[2]*Sbgr[2] - only_sky_pixels[...,2])**2
            + lambda_ * (x[2]-1.)**2
        ).sum()

    X = np.array( [0.15*np.ones( [1, only_sky_pixels.shape[0]] ),
                   210 *np.ones( [1, only_sky_pixels.shape[0]] ),
                   1.05*np.ones( [1, only_sky_pixels.shape[0]] )] )
    # projected gradient descent
    maxiter = 1000;
    alpha = 0.2*1e-5
    i=0
    while (i<maxiter):
        i=i+1
        grad = alpha*J_prime(X)
        projected_grad = grad
        # 0<=Alpha<=1
        projected_grad[0][(X-grad)[0] <=0]=0
        projected_grad[0][(X-grad)[0] >=1]=0
        # projected_grad[0][(X-grad)[0] <=0]=0
        X -= projected_grad
        # if not (i % 100):
        # print 'iter:', i, J(X)/1e5
        # if not (i%1000):
            # print X
            # if i == maxiter-1:
            # ipdb.set_trace()

    return X

def sky_enhance(X, P, Sbgr):
    """
    ARGs:
    -----
    X: Alpha = x[0]; C = x[1]; S=x[2]
    P: sky_porb_map
    Sbgr: np.array([[[Sb, Sg, Sr]]])

    NOTE: computing some color-like array (e.g. beta_old) will actually exceed 255,
          use 'float' instead of 'uint8'
    """
    def _mul(a, b):
        """
        produce mul res for things like 's*(Sb, Sg, Sr)'
        """
        # assert a.dtype == b.dtype
        res = np.empty( (b.shape[0], b.shape[1], 3) )
        res[...,0] = a[...,0]*b
        res[...,1] = a[...,1]*b
        res[...,2]  =a[...,2]*b
        return res

    def _2to3(A):
        """
        dimension increase [1,2] to [[1,1,1], [2,2,2]]
        extend A=arraytype(m,n) to (m,n,3), copy values
        m=1 in this func (sky_enhance)
        """
        res = np.empty([A.shape[0], A.shape[1], 3])
        res[:,:,0] = A; res[:,:,1] = A; res[:,:,2] = A
        return res

    # _h,_w = P.shape # original image shape
    # ipdb.set_trace()
    Alpha = X[0]; C = X[1]; S=X[2]
    f_sky_bgr = np.array([[[254,170,112]]], dtype='uint8')
    f_sky = cv2.cvtColor(f_sky_bgr, cv2.COLOR_BGR2LAB)
    beta_old = _2to3(S) * cv2.cvtColor(Sbgr, cv2.COLOR_BGR2LAB)
    kai_old = _2to3(C)
    W = np.array([[[100, 0, 0]]])

    P_3 = _2to3(P)
    beta_new = P_3*(f_sky*beta_old) + (1-P_3)*beta_old
    kai_new = P_3*(W+kai_old)/2.0 + (1-P_3)*kai_old
    res = _2to3(1-Alpha)*cv2.cvtColor(beta_new.astype('uint8'), cv2.COLOR_LAB2BGR) + _2to3(Alpha)*cv2.cvtColor(kai_new.astype('uint8'), cv2.COLOR_LAB2BGR)
    res = res.astype('uint8')
    return cv2.cvtColor(res, cv2.COLOR_LAB2BGR)

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    S, sky_prob_map = sky_ref_patch_detection(I_origin)
    #!! sky_prob_map should have 0~1 values, treat as 0?1?
    # ipdb.set_trace()
    X = sky_cloud_decompose(S, sky_prob_map, cape_util.mask_skin(I_origin, sky_prob_map.astype(bool)))
    Sbgr = np.array([[S]], dtype='uint8')
    P = sky_prob_map[sky_prob_to_01(sky_prob_map, thresh=0.0)].reshape(1,-1) # return sky_prob regarded as sky
    res = sky_enhance( X, P, Sbgr )
    res_disp = I_origin.copy();
    res_disp[sky_prob_to_01(sky_prob_map, thresh=0.0)] = res[0] # res.shape is (1, num_sky_pixels, 3)
    cape_util.display(np.hstack([I_origin, res_disp]))
    return 0

if __name__ == '__main__':
    main()
