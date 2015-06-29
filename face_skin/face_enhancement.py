"""
Created on June 18, 2015
@author: shiruilu

Face Enhancement from CAPE, put together
"""
# for search path
import os
import sys

source_dirs = ['cape_util', 'aindane', 'face_detection', 'skin_detection', 'wls_filter']

for d in source_dirs:
    sys.path.insert( 0, os.path.join(os.getcwd(), '../'+d) )

import cape_util
import aindane
import appendixa_skin_detect as apa_skin
import wls_filter
import docs_face_detector as vj_face

import cv2
import numpy as np
from matplotlib import pyplot as plt
# from sklearn import mixture
# from sklearn import cluster
from scipy.signal import argrelextrema


IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    _I_LAB = cv2.cvtColor(I_origin, cv2.COLOR_BGR2LAB)
    # WLS filter, only apply to L channel
    I = _I_LAB[...,0]
    Base, Detail = wls_filter.wlsfilter(I)
    I_out = Base
    I_aindane = aindane.aindane(I_origin)
    faces_xywh = vj_face.face_detect(I_aindane)
    faces = [ I_origin[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ]
    for face in faces:
        cv2.imwrite('teaser_face.png', face)
    skin_masks = [ apa_skin.skin_detect(face) for face in faces ]
    # for (skin, mask) in skin_masks:
        # cape_util.display( skin )

    _I_out_faces = [ I_out[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ]
    S = [ cv2.bitwise_and(_I_out_faces[i], skin_masks[i][1]) \
               for i in range(len(_I_out_faces)) ]
    for s in S:
        cape_util.display( s, mode='gray')
    _H = [ cv2.calcHist([s],[0],None,[255],[1,256]).T.ravel() for s in S ] #cv2.calcHist return 2d vector
    # smooth hist, correlate only take 1d input
    # http://stackoverflow.com/a/13730506/2729100
    H = [ np.correlate(_h, cv2.getGaussianKernel(30,10).ravel(), 'same') \
          for _h in _H ]
    # for s in S:
        # plt.hist(s.ravel(), 255, [1,256]); plt.xlim([1,256]); plt.show() # don't include 0(masked)
    for i in range(len(S)):
        # http://scikit-learn.org/0.10/modules/generated/sklearn.mixture.GMM.html
        # test with gmm
        # gmm_2 = mixture.GMM(n_components=2)
        # gmm_2.fit( h.reshape(len(h),1) )
        # print gmm_2.means
        # test with k-means
        # _kmeans = cluster.KMeans(k=3)
        # _kmeans.fit( S[i].reshape(-1,1) )
        # print _kmeans.cluster_centers_

        # http://stackoverflow.com/a/13491866/2729100
        # order: how many data points to consider on each side:
        # http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.argrelextrema.html
        maximas = argrelextrema(H[i], np.greater, order=10)
        minimas = argrelextrema(H[i], np.less, order=10)
        print "maximas: ", maximas, "minimas: ", minimas
        plt.plot(H[i]); plt.xlim([1,256]); plt.show()
    return 0

if __name__ == '__main__':
    main()