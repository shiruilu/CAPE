"""
Created on May 27, 2015
@author: shiruilu

skin detect from Appendix A of CAPE
"""

import numpy as np
import cv2
import time

def ellipse_test(A, B, bound=1.0):
    '''test a CIELab color falls in certain ellipse'''
    return (1.0*(A-143)/6.5)**2 + (1.0*(B-148)/12)**2 < bound

def check_neighbor(mask, i, j):
    '''for relaxed 4-neighbor test'''
    width, height = mask.shape
    return mask[i-1,j] + mask[(i+1)%width,j] + mask[i,j-1] + mask[i,(j+1)%height] >= 255

def HSV_threshold(H, S):
    return not(S>=0.25*255 and S<=0.75*255 or H>0.095*180) #s:0.25~0.75, h>0.095

def skin_detect(img_path):
    img = cv2.imread(img_path)
    # initialized all-white mask
    skinMask = 255*np.ones(img.shape[0:2], img.dtype)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    s = time.time()
    # ellipse and HSV_test
    for (i,j), value in np.ndenumerate(skinMask):
        skinMask[i,j] = 255 if ellipse_test(img_LAB[i,j][1], img_LAB[i,j][2]) and HSV_threshold(img_HSV[i,j][0], img_HSV[i,j][1]) else 0

    print time.time()-s
    # relaxed ellipse test, guarenteed by skin neighborhood
    for (i,j), value in np.ndenumerate(skinMask):
        if skinMask[i,j] ==0:
            skinMask[i,j] = 255 if ellipse_test(img_LAB[i,j][1], img_LAB[i,j][2], 1.25) and check_neighbor(skinMask, i, j) else 0

    print time.time()-s
    skin = cv2.bitwise_and(img, img, mask = skinMask)
    cv2.imshow('skin detect image', np.hstack([img, skin]))
    cv2.waitKey(0)
    cv2.imwrite('./benchmarks/ellipse_test_and_relaxed_ellipse.png', np.hstack([img, skin]))
    cv2.destroyAllWindows()

def test_ell():
    print ellipse_test(143, 148) # true, center
    print ellipse_test(143, 160) # edge case, false

def main():
    #test_ell()
    #skin_detect('./images/tiny_face.png')
    skin_detect('./images/input_teaser.png')
    return 0

if __name__ == '__main__':
    main()