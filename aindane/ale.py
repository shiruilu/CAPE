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

def ale(img_path):
    """ale algorithm in 3.1 of the paper"""
    I = cv2.imread(img_path)
    #?cv2 doing same as NTSC I_NTSC = (76.245*I[..., 2] + 149.685*I[..., 1] + 29.07*I[..., 0])/255.0 # equation 1
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    In = I/255.0 #2d array, equation 2
    #hist, bins = np.histogram(I.flatten(), bins=256, range=[0,256])
    hist = cv2.calcHist([I],[0],None,[256],[0,256])
    cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max()/ cdf.max()
    L = np.searchsorted(cdf, 0.1*I.shape[0]*I.shape[1], side='right') # http://stackoverflow.com/a/25032853/2729100
    La = np.array([L]) # L as array
    z = np.piecewise(La, [La<=50, La>50 and La<=150, La>150], [0, (L-50)/100.0, 1]) #http://stackoverflow.com/a/16575522/2729100

    In_prime = (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z)) /2.0
    return 0

def ace(img_path, c=5):
    """"ace algo in 3.2"""
    # https://youtu.be/jWVCHZUfbyY?t=4m8s, sigma and c
    sigma = sqrt(c**2 /2)
    ## K = 1/(sigma * sqrt(2*pi)) # once forget, see black pen notes on paper
    # transform to freqency domain: http://stackoverflow.com/questions/12861641/using-fft2-with-reshaping-for-an-rgb-filter
    image = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_freq = np.fft.fft2(image)
    img_freq_shift = np.fft.fftshift(img_freq)
    # decide on size of Gaussian filter: http://stackoverflow.com/questions/16165666/how-to-determine-the-window-size-of-a-gaussian-filter
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    gaussian_x = cv2.getGaussianKernel(int(round(sigma*3)), int(round(sigma))) ##CAUTION, sigma is int??
    gaussian = gaussian_x*gaussian_x.T
    gaussian_freq_shift = np.fft.fftshift( np.fft.fft2(gaussian, image.shape) ) ##gaussian kernel padded with 0, extend to image.shape

    # from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    image_fm = img_freq_shift * gaussian_freq_shift # element wise multiplication
    image_ff = np.real( np.fft.ifft2( np.fft.ifftshift(image_fm) ) ) # from http://kitchingroup.cheme.cmu.edu/pycse/pycse.html

    plt.imshow(np.hstack([image, image_ff]), 'gray')
    plt.show()
    return 0

def test_ale():
    ale(IMG_DIR+'pdbuse.png')
    return 0

def test_ace():
    ace(IMG_DIR+'input_teaser.png')
    return 0

if __name__ == '__main__':
    #test_ale()
    test_ace()