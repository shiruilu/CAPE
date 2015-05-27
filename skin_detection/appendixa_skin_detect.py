"""
Created on May 27, 2015
@author: shiruilu

skin detect from Appendix A of CAPE
"""

import numpy as np
import cv2

def ellipse_test(A, B):
    return (1.0*(A-143)/6.5)**2 + (1.0*(B-148)/12)**2 < 1

def skin_detect(img_path):
    img = cv2.imread(img_path)
    # initialized all-white mask
    skinMask = 255*np.ones(img.shape[0:2], img.dtype)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    for (i,j), value in np.ndenumerate(skinMask):
        skinMask[i,j] = 255 if ellipse_test(img_LAB[i,j][1], img_LAB[i,j][2]) else 0

    skin = cv2.bitwise_and(img, img, mask = skinMask)
    cv2.imshow('skin detect image', np.hstack([img, skin]))
    cv2.waitKey(0)
    cv2.imwrite('./benchmarks/ellpse_test_only.png', np.hstack([img, skin]))
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