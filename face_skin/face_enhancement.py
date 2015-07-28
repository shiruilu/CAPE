"""
Created on June 18, 2015
@author: shiruilu

Face Enhancement from CAPE, put together
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

from eacp import EACP

import cv2
import numpy as np
from matplotlib import pyplot as plt
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def sidelight_correction(I_L, I_out, H, S, faces_xywh, _eacp_lambda_):
    I_out_255 = cape_util.mag(I_out, 'float')
    bimodal_Fs, D, M, B = cape_util.detect_bimodal(H)
    A = np.ones(I_out_255.shape)
    W = np.zeros(I_out_255.shape)
    for i in range(len(bimodal_Fs)):
        if bimodal_Fs[i] == True:
            b = B[i]; d = D[i]; m = M[i];
            f = (b-d) / (m-d)
            print str(i)+'th face: d, m, b, f', d,m,b,f
            x,y,w,h = faces_xywh[i]
            A[y:y+h, x:x+w][(cape_util.mag(S[i], 'float') <m)
                        & (cape_util.mag(S[i], 'float') >0)] = f #<m and \in S
            global W
            miu = (I_out_255[A==f]).mean()
            sig = 255*3 # manually set, 3*sig = 120 close to 255/2
            W = np.exp(-(I_out_255-miu)**2/sig**2)
            # W[A==f] = 1.
            W[...] = 1-W[...]
            # W[A!=f] =1.
            W[...] = 1.
            # face_i_crr = EACP(I_out_255[y:y+h,x:x+w]*A, I_out_255[y:y+h,x:x+w], lambda_=_eacp_lambda_)
            # I_out_255[y:y+h, x:x+w] = face_i_crr
        else:
            print '? bimodal not detected on', i, 'th face'
    # W[...] = 1.
    cape_util.display(W, mode='rainbow')
    A_after = EACP(A, I_L, W, lambda_=_eacp_lambda_)
    I_out_side_crr = A_after * I_out_255
    # to visualize A before/after EACP
    cape_util.display( np.hstack([A, A_after]), mode='rainbow' )
    # to visualize sidelight corrected result
    cape_util.display( cape_util.mag(I_out_side_crr, 'trim')
                       , name='sidelight corrected, L' ,mode='gray')
    return I_out_side_crr # float [0.0,255.0]

def exposure_correction(I_L, I_out, I_out_side_crr, skin_masks, faces_xywh
                        , _eacp_lambda_):
    I_out_255 = cape_util.mag(I_out, 'float')
    A = np.ones(I_out_side_crr.shape)
    I_out_expo_crr = I_out_side_crr.copy()
    for i in range(len(faces_xywh)):
        x,y,w,h = faces_xywh[i]
        face = I_out_side_crr[y:y+h, x:x+w]; # each sidelit crted face(L channel)
        skin_face = cape_util.mask_skin(face, skin_masks[i][1]) # SKIN on that face
        cumsum = cv2.calcHist([cape_util.mag(skin_face, 'trim')] # cumsumed hist of SKIN
                              ,[0],None,[255],[1,256]).T.ravel().cumsum()
        # import ipdb; ipdb.set_trace()
        # visualize cumsum of hist
        plt.plot(cumsum); plt.xlim([1,256]); plt.show()

        p = np.searchsorted(cumsum, cumsum[-1]*0.75)
        if p <120:
            print 'face skin is underexposed!'
            f = (120+p)/((2*p)+1e-6)
            A[y:y+h, x:x+w][face >0] = f; # >0 means every pixel *\in S*
            A_after = EACP(A, I_out_255, lambda_=_eacp_lambda_)
            # to visualize A before/after EACP
            cape_util.display( np.hstack([A, A_after]), mode='rainbow' )
            I_out_expo_crr = A_after * I_out_side_crr
    # to visualize exposure corrected face
    cape_util.display( cape_util.mag(I_out_expo_crr, 'trim')
                       , name='exposure corrected L', mode='gray')
    return I_out_expo_crr

def face_enhancement(I_origin, _eacp_lambda_=0.2):
    """
    Keyword Arguments:
    I             -- np.uint8, (m,n,3), BGR
    _eacp_lambda_ -- float, scala
    """
    _I_LAB = cv2.cvtColor(I_origin, cv2.COLOR_BGR2LAB)
    I = _I_LAB[...,0]
    Base, Detail = wls_filter.wlsfilter(I)
    I_out = Base # float [0,1]

    I_aindane = aindane.aindane(I_origin)
    faces_xywh = vj_face.face_detect(I_aindane)
    # import ipdb; ipdb.set_trace()
    faces = [ I_origin[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ]
    skin_masks = [ apa_skin.skin_detect(face) for face in faces ]
    _any_skin = False
    for _mask in skin_masks:
        _any_skin |= ((_mask[1]).any())
    if (faces_xywh==[]) or (not _any_skin): # face not detected
        print 'face or skin not detected!'
        return I_origin
    _I_out_faces = [ I_out[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ] #float[0,1]
    S = [ cape_util.mask_skin(_I_out_faces[i], skin_masks[i][1]) \
          for i in range(len(_I_out_faces)) ] # float [0,1]
    
    # to visualize detected skin and it's (unsmoothed) hist
    for s in S:
        cape_util.display( cape_util.mag(s)
                           , name='detected skin of L channel', mode='gray')
        # plot original hist(rectangles form, of S). don't include 0(masked)
        plt.hist(cape_util.mag(s).ravel(), 255, [1,256])
        plt.xlim([1,256]); plt.show()
        
    # unsmoothed hist (cv2.calcHist return 2d vector)
    _H = [ cv2.calcHist([cape_util.mag(s)],[0],None,[255],[1,256]).T.ravel()
           for s in S ]
    # smooth hist, correlate only takes 1d input
    H = [ np.correlate(_h, cv2.getGaussianKernel(30,10).ravel(), 'same')
          for _h in _H ]
    
    # visualize smoothed hist
    for h in H:
        plt.plot(h); plt.xlim([1,256]); plt.show()
    
    I_out_side_crr = sidelight_correction(_I_LAB[...,0], I_out, H, S, faces_xywh, _eacp_lambda_)
    I_out_expo_crr = exposure_correction(_I_LAB[...,0], I_out, I_out_side_crr, skin_masks
                                         , faces_xywh, _eacp_lambda_)

    _I_LAB[...,0] = cape_util.mag( (I_out_expo_crr + 255.0*Detail), 'trim' )
    I_res = cv2.cvtColor(_I_LAB, cv2.COLOR_LAB2BGR)
    return I_res

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    lambda_ = 120
    I_res = face_enhancement(I_origin, lambda_)
    cape_util.display( np.hstack([I_origin, I_res]) )
    return 0

if __name__ == '__main__':
    main()