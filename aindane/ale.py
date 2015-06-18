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
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)
    #?cv2 doing same as NTSC I_NTSC = (76.245*I[..., 2] + 149.685*I[..., 1] + 29.07*I[..., 0])/255.0 # equation 1
    In = I/255.0 #2d array, equation 2
    #hist, bins = np.histogram(I.flatten(), bins=256, range=[0,256])
    hist = cv2.calcHist([I],[0],None,[256],[0,256])
    cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max()/ cdf.max()
    L = np.searchsorted(cdf, 0.1*I.shape[0]*I.shape[1], side='right') # http://stackoverflow.com/a/25032853/2729100
    L_as_array = np.array([L]) # L as array
    z = np.piecewise(L_as_array, [L_as_array<=50, L_as_array>50 and L_as_array<=150, L_as_array>150], [0, (L-50)/100.0, 1]) #http://stackoverflow.com/a/16575522/2729100

    In_prime = (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z)) /2.0
    return I, In_prime

def ace(I, In_prime, c=5):
    """"ace algo in 3.2"""
    # https://youtu.be/jWVCHZUfbyY?t=4m8s, sigma and c
    sigma = sqrt(c**2 /2)
    ## K = 1/(sigma * sqrt(2*pi)) # once forget, see black pen notes on paper
    # transform to freqency domain: http://stackoverflow.com/questions/12861641/using-fft2-with-reshaping-for-an-rgb-filter
    #I = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_freq = np.fft.fft2(I)
    img_freq_shift = np.fft.fftshift(img_freq)
    # decide on size of Gaussian filter: http://stackoverflow.com/questions/16165666/how-to-determine-the-window-size-of-a-gaussian-filter
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    gaussian_x = cv2.getGaussianKernel(int(round(sigma*3)), int(round(sigma))) ##CAUTION, sigma is int??
    gaussian = gaussian_x*gaussian_x.T
    gaussian = gaussian / np.sum(gaussian)
    gaussian_freq_shift = np.fft.fftshift( np.fft.fft2(gaussian, I.shape) ) ##gaussian kernel padded with 0, extend to image.shape

    # from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    image_fm = img_freq_shift * gaussian_freq_shift # element wise multiplication
    # from http://kitchingroup.cheme.cmu.edu/pycse/pycse.html
    I_conv = np.real( np.fft.ifft2( np.fft.ifftshift(image_fm) ) ) # equation 6

    sigma_I = np.std(I) # http://stackoverflow.com/a/13411610/2729100
    sigma_I = np.array([sigma_I]) # to an array
    P = np.piecewise(sigma_I, [sigma_I<=3, sigma_I>3 and sigma_I<10, sigma_I>=10], [3, 1.0*(27-2*sigma_I)/7, 1]) #http://stackoverflow.com/a/16575522/2729100
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
    # S_20 = ace(I, In_prime, c=20)
    # S_240 = ace(I, In_prime, c=240)
    # restore using color_restoration (aindane paper)
    S_restore = color_restoration(I_bgr, I, S, [1,1,1]) # choose default lambda as all 1s
    S_display = cv2.cvtColor( np.clip(S_restore, 0, 255).astype('uint8'), cv2.COLOR_BGR2RGB)
    I_rgb = cv2.cvtColor( I_bgr, cv2.COLOR_BGR2RGB)

    # restore using opencv
    #S_display = cv2.cvtColor(np.clip(S, 0, 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

    # plt.imshow( np.hstack([I, S, S_20, S_240]), 'gray' )
    plt.imshow( np.hstack([I_rgb, S_display]) )
    plt.show()

if __name__ == '__main__':
    _test_all()