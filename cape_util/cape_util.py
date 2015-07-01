"""
Created on June 18, 2015
@author: shiruilu

Common utils for CAPE
"""

import cv2
import matplotlib.pyplot as plt

def mask_skin(img, mask):
    img_cp = img.copy()
    img_cp[ ~mask ] = 0 # non-skin area set to 0
    return img_cp

def display(img, name='', mode='bgr'):
    """
    display image using matplotlib

    ARGS:
    img: bgr mode
    name: string, displayed as title
    """
    if mode == 'bgr':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif mode == 'rgb':
        plt.imshow(img)
    elif mode == 'gray':
        plt.imshow(img, 'gray')
    else:
        raise ValueError('CAPE display: unkown mode')
    plt.title(name)
    plt.show()