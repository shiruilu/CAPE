"""
Created on June 9, 2015
@author: shiruilu

Adaptive Luminance Enhancement from AINDANE
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt, pi
import time

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'
eps = 1e-6 # eliminate divide by zero error in I_conv/I

def ale(I_bgr):
    """ale algorithm in 3.1 of the paper"""
    #?cv2 doing same as NTSC # equation 1
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)
    In = I/255.0 #2d array, equation 2
    hist = cv2.calcHist([I],[0],None,[256],[0,256])
    cdf = hist.cumsum()
    L = np.searchsorted(cdf, 0.1*I.shape[0]*I.shape[1], side='right')
    L_as_array = np.array([L]) # L as array, for np.piecewise
    z = np.piecewise( L_as_array,
                      [ L_as_array<=50,
                        L_as_array>50 and L_as_array<=150,
                        L_as_array>150
                      ],
                      [ 0, (L-50)/100.0, 1 ]
    )

    In_prime = (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z)) /2.0
    return I, In_prime

def ace(I, In_prime, c=5):
    """"ace algo in 3.2"""
    sigma = sqrt(c**2 /2)
    img_freq = np.fft.fft2(I)
    img_freq_shift = np.fft.fftshift(img_freq)
    # size of gaussian: 3*sigma(0.99...), cv2 require sigma to be int
    _gaussian_x = cv2.getGaussianKernel(int(round(sigma*3)), int(round(sigma)))
    gaussian = (_gaussian_x * _gaussian_x.T) / np.sum(_gaussian_x * _gaussian_x.T) # normalize
    ##gaussian kernel padded with 0, extend to image.shape
    gaussian_freq_shift = np.fft.fftshift( np.fft.fft2(gaussian, I.shape) )

    image_fm = img_freq_shift * gaussian_freq_shift # element wise multiplication
    I_conv = np.real( np.fft.ifft2( np.fft.ifftshift(image_fm) ) ) # equation 6

    sigma_I = np.array( np.std(I) ) # std of I,to an array, for np.piecewise
    P = np.piecewise(sigma_I,
                     [ sigma_I<=3,
                       sigma_I>3 and sigma_I<10,
                       sigma_I>=10
                     ],
                     [ 3, 1.0*(27-2*sigma_I)/7, 1 ]
    )
    E = ((I_conv+eps) / (I+eps)) ** P
    S = 255 * np.power(In_prime, E)
    return S

def color_restoration(I_bgr, I, S, lambdaa):
    S_restore = np.zeros(I_bgr.shape)
    for j in range(3): # b,g,r
        S_restore[...,j] = S * (1.0* I_bgr[...,j] / I) * lambdaa[j]
    return S_restore

def _test_ale():
    ale(IMG_DIR+'pdbuse.png')
    return 0

def _test_ace():
    ace(IMG_DIR+'input_teaser.png')
    return 0

def _test_color_restoration():
    I_rgb = cv2.cvtColor(cv2.imread(IMG_DIR+'input_teaser.png'), cv2.COLOR_BGR2RGB)
    I = cv2.cvtColor(I_rgb, cv2.COLOR_RGB2GRAY)
    color_restoration(I,I)

def _test_all():
    I_bgr = cv2.imread(IMG_DIR+'input_teaser.png')
    I, In_prime = ale(I_bgr)
    S = ace(I, In_prime, c=240)
    # restore using color_restoration (aindane paper)
    S_restore = color_restoration(I_bgr, I, S, [1,1,1]) # choose default lambda as all 1s
    S_display = cv2.cvtColor( np.clip(S_restore, 0, 255).astype('uint8'), cv2.COLOR_BGR2RGB)
    I_rgb = cv2.cvtColor( I_bgr, cv2.COLOR_BGR2RGB)

    plt.imshow( np.hstack([I_rgb, S_display]) )
    plt.show()

def aindane(I_bgr):
    I, In_prime = ale(I_bgr)
    S = ace(I, In_prime, c=240)
    S_restore = color_restoration(I_bgr, I, S, [1,1,1]) # choose default lambda as all 1s
    S_bgr = np.clip(S_restore, 0, 255).astype('uint8')
    return S_bgr

if __name__ == '__main__':
    _test_all()