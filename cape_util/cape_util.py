"""
Created on June 18, 2015
@author: shiruilu

Common utils for CAPE
"""

import cv2
import matplotlib.pyplot as plt


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