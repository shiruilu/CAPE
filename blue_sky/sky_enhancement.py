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

import cv2
import numpy as np
from scipy.stats import norm
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import ipdb

# # tricky way of removing ipdb breakpoints
# def f(): pass
# ipdb.set_trace = f

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def in_idea_blue_rg(I_origin):
    img = I_origin.astype(float)
    IDEAL_SKY_BGR = (195, 165, 120)
    # RANGE = (25, 30, 50)
    RANGE = (60, 65, 80)
    return (abs(img[...,0]-IDEAL_SKY_BGR[0]) < RANGE[0]) \
         & (abs(img[...,1]-IDEAL_SKY_BGR[1]) < RANGE[1]) \
         & (abs(img[...,2]-IDEAL_SKY_BGR[2]) < RANGE[2])

def _var_lenpos(val, pos):
    '''
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/ndimage/measurements.py#L330
    '''
    return val, len(pos), pos

def get_sky_ref_patch(f_gray_one_third, sky_prob_map):
    """
    Return mean and std of the *largest* sky patch detected on the input image
    +TODO: clear patches other than the largest (set sky_prob to 0)
    Will MODIFY sky_prob_map
    """
    lbl, nlbl = ndi.label(f_gray_one_third)
    lbls = np.arange(1, nlbl+1)
    _res = ndi.labeled_comprehension(f_gray_one_third, lbl, lbls, _var_lenpos, object, 0, True)

    val = []; lenpos = []; pos_arr = []
    for idx, (_val, _lenpos, _pos) in np.ndenumerate(_res):
        val.append(_val); lenpos.append(_lenpos); pos_arr.append(_pos)

    pos_am = np.array(lenpos).argmax() # pos of largest sky ref patch
    print 'max patch amount, patch: ', lenpos[pos_am], val[pos_am]
    mean = np.mean(val[pos_am])
    std = np.std(val[pos_am])

    # clear patches other than the largest (set sky_prob to 0)
    # import ipdb; ipdb.set_trace()
    for i, _p in enumerate(pos_arr):
        if i != pos_am:
            sky_prob_map.ravel()[_p] = 0.
    return mean, std

def _get_top_one_third(img):
    return img[0:img.shape[0]*1.0/3,...]

def sky_prob_to_01(sky_prob_map, thresh=0.0):
    return sky_prob_map > thresh

def _2to3(A):
    """
    dimension increase [1,2] to [[1,1,1], [2,2,2]]
    extend A=arraytype(m,n) to (m,n,3), copy values
    m=1 in this func (sky_enhance)
    """
    res = np.empty([A.shape[0], A.shape[1], 3])
    res[:,:,0] = A; res[:,:,1] = A; res[:,:,2] = A
    return res

def sky_ref_patch_detection(I_origin):
    """
    RETURN:
    S: list:[Sb, Sg, Sr]
    sky_prob_map
    """
    I_gray = cv2.cvtColor(I_origin, cv2.COLOR_BGR2GRAY)
    sky_prob_map = np.zeros(I_gray.shape)
    # initialize sky prob map with pixels strictly within ideal blue range
    sky_prob_map[ in_idea_blue_rg(I_origin) ] = 1.0
    # sky is ONLY in top one third
    # sky_prob_map[sky_prob_map.shape[0]*1.0/3:-1,...] = 0.0

    # exponentially decrease sky prob where gradient is too large (>_grad_percent*255.)
    _grad_x = np.absolute( ndi.prewitt(I_gray, axis=1 ,mode='nearest') )
    _grad_y = np.absolute( ndi.prewitt(I_gray, axis=0 ,mode='nearest') )
    _rv = norm(loc=0., scale=255./3.) # 3*sigma = 255, rv ~ random variable

    _grad_percent = 0.10 # 0.05 of the original paper
    cond_mod_sky_prob = (sky_prob_map ==1.0) & \
                        ((_grad_x >_grad_percent*255.) | (_grad_y >_grad_percent*255.)) # average of gradx, y
    sky_prob_map[ cond_mod_sky_prob ] = _rv.pdf( (_grad_x + _grad_y)/2.0 )[ cond_mod_sky_prob ]

    # detect bimodal
    _L = cv2.cvtColor(I_origin, cv2.COLOR_BGR2LAB)[...,0]
    detect_res = cape_util.detect_bimodal(
        [cape_util.get_smoothed_hist( cape_util.mask_skin(_L, sky_prob_map!=0.0) )]
    )[0] # detect_bimodal is an array f
    if (detect_res[0] == True): #could return F or None, must be ==True
        print 'bimodal detected in current sky_ref_patch'
        sky_prob_map[ I_gray ==detect_res[1] ] = 0.0 #exclude pixels correspond to the dark mode

    # get mean and std from each b,g,r channel of detected sky
    S = []
    if ( np.sum( _get_top_one_third(sky_prob_map) ) !=0): # top 1/3 has sky
        sky_prob_map_bgr = _2to3(sky_prob_map)
        for i in range(3): #B, G, R
            masked_I_one_thrid = cape_util.mask_skin( # mask non-blue area 0, copy the rest
                _get_top_one_third(I_origin)[...,i]
                , _get_top_one_third(sky_prob_map)!=0.0 # sky_prob map changed after previous step
            )
            mean, std = get_sky_ref_patch( masked_I_one_thrid, sky_prob_map )
            S.append(mean)
            _rv_sky_patch = norm(loc=mean, scale=std)

            # re-assign (where sky prob>0), normalize to p(median) = 1.0
            sky_prob_map_bgr[...,i][sky_prob_map>0.0] = \
                _rv_sky_patch.pdf(I_origin[...,i])[sky_prob_map>0.0] / _rv_sky_patch.pdf(mean)
        _b=1.; _g=5.; _r=3.
        sky_prob_map = (_b*sky_prob_map_bgr[...,0] + _g*sky_prob_map_bgr[...,1] + _r*sky_prob_map_bgr[...,2]) / (_b+_g+_r)

        # remove small patches: image opening on sky_prob_map
        _h,_w = sky_prob_map.shape[0:2]
        _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(_h/12,_w/16))
        sky_mask_opened = cv2.morphologyEx(sky_prob_to_01(sky_prob_map).astype('uint8'), cv2.MORPH_OPEN, _kernel)
        sky_prob_map[sky_mask_opened==0.0] = 0.0 # set sky_prob_map to 0 where removed from sky_mask_opened
    else:
        print 'sky not detected in top 1/3'

    plt.imshow(sky_prob_map); plt.show() # rainbow map
    print 'S(b,g,r): ',S
    return S, sky_prob_map

def sky_cloud_decompose(Sbgr, sky_prob_map ,sky_pixels, lambda_=1.0):

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
            ( x[0]*x[1] + (1-x[0])*x[2]*Sbgr[0] - only_sky_pixels[...,0])**2
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
        # 0<=Alpha<=1, method 1
        projected_grad[0][(X-grad)[0] <=0]=0
        projected_grad[0][(X-grad)[0] >=1]=0
        X -= projected_grad

        if not (i % 100):
            print 'iter:', i, J(X)/1e5
    # visualize Alpha
    _sky_cloud_prob = sky_prob_to_01(sky_prob_map, thresh=0.0).astype('float64') # black & white
    _sky_cloud_prob[ sky_prob_to_01(sky_prob_map, thresh=0.0) ] = X[0][0]
    plt.imshow( np.hstack([sky_prob_map, _sky_cloud_prob] ), cmap=plt.cm.rainbow); plt.show()
    return X

def sky_enhance(X, P, Sbgr
                ,I_origin, sky_prob_map):
    """
    ARGs:
    -----
    X: Alpha = x[0]; C = x[1]; S=x[2]
    P: sky_porb_map
    Sbgr: np.array([[[Sb, Sg, Sr]]]), mean of the sky color in current image

    NOTE: computing some color-like array (e.g. beta_old) will actually exceed 255,
          use 'float' instead of 'uint8'
    """
    Alpha = X[0]; C = X[1]; S=X[2]

    f_sky_bgr = np.array([[[255,210,160]]], dtype='uint8')
    f_sky = cv2.cvtColor(f_sky_bgr, cv2.COLOR_BGR2LAB) # preferred color, in CIELab
    Slab = cv2.cvtColor(Sbgr, cv2.COLOR_BGR2LAB)
    f_lab = 1.0*f_sky / Slab # (f_l, f_a, f_b), correction ratio
    print 'f_lab: ', f_lab
    beta_old = cv2.cvtColor( cape_util.safe_convert(_2to3(S) * Sbgr, np.uint8), cv2.COLOR_BGR2LAB )
    kai_old = cv2.cvtColor( cape_util.safe_convert(_2to3(C), np.uint8), cv2.COLOR_BGR2LAB )
    W = np.array([[[100, 0, 0]]])
    # W = np.array([[[255, 128, 128]]]) # (255,255,255) in RGB

    P_3 = _2to3(P)
    beta_new = cv2.add(P_3*(f_lab*beta_old), (1-P_3)*beta_old)
    kai_new = cv2.add(P_3*((W+kai_old)/2.0), (1-P_3)*kai_old)
    _fst = _2to3(1-Alpha)*cv2.cvtColor(beta_new.astype('uint8'), cv2.COLOR_LAB2BGR)
    _lst = _2to3(Alpha)*cv2.cvtColor(kai_new.astype('uint8'), cv2.COLOR_LAB2BGR)
    res = cv2.add( _fst, _lst ) # use cv2.add() to avoid overflow e.g. 150+125=255, not 20
    # visualize _fst, _lst for debugging
    _first = I_origin.copy(); _last = I_origin.copy()
    _first[ sky_prob_to_01(sky_prob_map, thresh=0.0) ] = _fst[0]
    _last[ sky_prob_to_01(sky_prob_map, thresh=0.0) ] = _lst[0]
    cape_util.display( np.hstack([_first, _last] ) )

    return res

def sky_enhancement(I):
    """
    Main function to perform sky enhancement
    Keyword Arguments:
    I -- np.uint8 (m,n,3), BGR
    Returns:
    res -- np.uint8 (m,n,3), I with sky part enhanced
    sky_prob_map -- np.float 0-1 (m,n,1), sky probability map
    """
    S, sky_prob_map = sky_ref_patch_detection(I)
    res = I.copy() # res shape is (1, num_sky_pixels, 3)
    if S == []:
        print 'S is empty'
    elif sky_prob_to_01(sky_prob_map, thresh=0.).any():
        X = sky_cloud_decompose(S, sky_prob_map, cape_util.mask_skin(I, sky_prob_map.astype(bool)))
        Sbgr = np.array([[S]], dtype='uint8')
        P = sky_prob_map[sky_prob_to_01(sky_prob_map, thresh=0.0)].reshape(1,-1) # return sky_prob regarded as sky
        enhance_res = sky_enhance( X, P, Sbgr, I, sky_prob_map)
        res[sky_prob_to_01(sky_prob_map, thresh=0.0)] = enhance_res[0]
    else:
        print 'no sky pixels detected!'

    return res, sky_prob_map

def main():
    I_origin = cv2.imread(IMG_DIR+'pic23.jpg')
    res_disp,_ = sky_enhancement(I_origin)
    cape_util.display(np.hstack([I_origin, res_disp]))
    return 0

if __name__ == '__main__':
    main()
