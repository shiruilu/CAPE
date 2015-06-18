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

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def main():
    I_bgr = cv2.imread(IMG_DIR+'input_teaser.png')
    I_aindane = aindane.aindane(I_bgr)
    # cape_util.display(I_aindane)
    faces_xywh, I_face_rectangle = vj_face.face_detect(I_aindane)
    cape_util.display(I_face_rectangle)
    return 0

if __name__ == '__main__':
    main()