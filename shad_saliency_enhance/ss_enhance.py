"""
Created on July 16, 2015
@author: shiruilu

Shadowed Saliency Enhancement
"""
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

import cv2
import numpy as np
from scipy.stats import norm
from scipy import ndimage as ndi
from scipy import optimize as opt
from matplotlib import pyplot as plt
import ipdb

# tricky way of removing ipdb breakpoints
def f(): pass
ipdb.set_trace = f

def main():
    return

if __name__ == '__main__':
    main()