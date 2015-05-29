"""
Created on May 26, 2015
@author: shiruilu

try out simple skin-detection from
http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
"""

import numpy as np
import cv2

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
IMG_DIR = '../resources/images/'

def skin_detect(img_path):
    img = cv2.imread(img_path)
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    '''
    # show skin with other parts masked
    skin = cv2.bitwise_and(img, img, mask = skinMask)
    cv2.imshow('sample image', np.hstack([img, skin]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    skin_detect(IMG_DIR+'tiny_face.png')
    return 0

if __name__ == '__main__':
    main()