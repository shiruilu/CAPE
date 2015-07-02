"""
Created on June 18, 2015
@author: shiruilu

Face Enhancement from CAPE, put together
"""
# for search path
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
from scipy.signal import argrelextrema
import ipdb

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def detect_bimodal(H):
    """
    H: all the (smoothed) histograms of faces on the image
    RETURN:
    bimodal_Fs: True means detected
                False means undetected (i.e. not bimodal)
                None means not sure, will plot H[i] for analysis
    """
    # argrelextrema return (array([ 54, 132]),) (a tuple), only [0] used for 1d
    maximas_Fs = [ argrelextrema(h, np.greater, order=10)[0] for h in H ]
    # argrelextrema return (array([ 54, 132]),) (a tuple), only [0] used for 1d
    minimas_Fs = [ argrelextrema(h, np.less, order=10)[0] for h in H ]
    # # to visualize the bimodal:
    # print "maximas each face(hist): ", maximas_Fs, "minimas each face(hist): ", minimas_Fs
    # plt.plot(H[i]); plt.xlim([1,256]); plt.show()
    bimodal_Fs = np.zeros(len(H) ,dtype=bool)
    D = np.zeros(len(H)); M = np.zeros(len(H)); B = np.zeros(len(H));
    for i in range(len(H)): # each face i
        tot_face_pix = np.sum(H[i])
        # pdb.set_trace()
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

def sidelight_correction(I_out, H, S, faces_xywh, _eacp_lambda_):
    I_out_255 = cape_util.mag(I_out, 'float')
    bimodal_Fs, D, M, B = detect_bimodal(H)
    A = np.ones(I_out_255.shape)
    for i in range(len(bimodal_Fs)):
        if bimodal_Fs[i] == True:
            b = B[i]; d = D[i]; m = M[i];
            f = (b-d) / (m-d)
            x,y,w,h = faces_xywh[i]
            # ipdb.set_trace()
            A[y:y+h, x:x+w][(cape_util.mag(S[i], 'float') <m) & (cape_util.mag(S[i], 'float') >0)] = f; # <m and \in S
        else:
            print '? bimodal not detected on', i, 'th face'

    I_out_side_crr = EACP(I_out_255 *A, I_out_255, lambda_=_eacp_lambda_)
    # cape_util.display( cape_util.mag(I_out_side_crr, 'trim'), name='sidelight corrected, L' ,mode='gray')
    return I_out_side_crr # float [0.0,255.0]

def exposure_correction(I_out, I_out_side_crr, skin_masks, faces_xywh, _eacp_lambda_):
    I_out_255 = cape_util.mag(I_out, 'float')
    A = np.ones(I_out_side_crr.shape)
    I_out_expo_crr = I_out_side_crr.copy()
    for i in range(len(faces_xywh)):
        x,y,w,h = faces_xywh[i]
        face = I_out_side_crr[y:y+h, x:x+w]; # each sidelight corrected face (L channel)
        skin_face = cape_util.mask_skin(face, skin_masks[i][1]) # skin on that face
        cumsum = cv2.calcHist([cape_util.mag(skin_face)],[0],None,[255],[1,256]).T.ravel().cumsum() # hist of the skin
        p = np.searchsorted(cumsum, cumsum[-1]*0.75)
        if p <120:
            f = (120+p)/(2*p)
            A[y:y+h, x:x+w][face >0] = f; # >0 means every pixel *\in S*
            I_out_expo_crr = EACP(A*I_out_side_crr, I_out_255, lambda_=_eacp_lambda_)

    # cape_util.display( cape_util.mag(I_out_expo_crr, 'trim'), name='exposure corrected L', mode='gray')
    return I_out_expo_crr

def CAPE(I_origin, _eacp_lambda_=0.2):
    _I_LAB = cv2.cvtColor(I_origin, cv2.COLOR_BGR2LAB)
    # WLS filter, only apply to L channel
    I = _I_LAB[...,0]
    Base, Detail = wls_filter.wlsfilter(I)
    I_out = Base # float [0,1]

    I_aindane = aindane.aindane(I_origin)
    faces_xywh = vj_face.face_detect(I_aindane)
    faces = [ I_origin[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ]
    skin_masks = [ apa_skin.skin_detect(face) for face in faces ]
    _I_out_faces = [ I_out[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ] # float [0,1]
    S = [ cape_util.mask_skin(_I_out_faces[i], skin_masks[i][1]) \
               for i in range(len(_I_out_faces)) ] # float [0,1]
    # for s in S:
    #     cape_util.display( cape_util.mag(s), name='detected skin of L channel', mode='gray')
    #     # plot original hist(rectangles form, of S). don't include 0(masked)
    #     plt.hist(cape_util.mag(s).ravel(), 255, [1,256]); plt.xlim([1,256]); plt.show()
    # unsmoothed hist (cv2.calcHist return 2d vector)
    _H = [ cv2.calcHist([cape_util.mag(s)],[0],None,[255],[1,256]).T.ravel() for s in S ]
    # smooth hist, correlate only take 1d input
    H = [ np.correlate(_h, cv2.getGaussianKernel(30,10).ravel(), 'same') \
          for _h in _H ]
    # # visualize smoothed hist
    # for h in H:
    #     plt.plot(h); plt.xlim([1,256]); plt.show()
    # ipdb.set_trace()
    I_out_side_crr = sidelight_correction(I_out, H, S, faces_xywh, _eacp_lambda_)
    I_out_expo_crr = exposure_correction(I_out, I_out_side_crr, skin_masks, faces_xywh, _eacp_lambda_)

    _I_LAB[...,0] = cape_util.mag( (I_out_expo_crr/255.0 + Detail) )
    I_res = cv2.cvtColor(_I_LAB, cv2.COLOR_LAB2BGR)
    return I_res
    # cape_util.display(np.hstack([I_origin, I_res]), name='final result, lambda= '+str(_eacp_lambda_), mode='bgr')
    # print I_origin == I_res
    # print (I_origin == I_res).all()

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    for lambda_ in cape_util.frange(0.2, 4.0, 0.2):
        I_res = CAPE(I_origin, lambda_)
        DIR = './eacp_lambda/'
        cv2.imwrite( DIR+'eacp_lambda='+str(lambda_)+'.png', np.hstack([I_origin, I_res]) )
    return 0

if __name__ == '__main__':
    main()