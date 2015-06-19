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
import numpy as np
import copy

IMG_DIR = '../resources/images/'
BM_DIR = './benchmarks/'

def main():
    I_origin = cv2.imread(IMG_DIR+'input_teaser.png')
    I_bgr = copy.deepcopy(I_origin)
    I_aindane = aindane.aindane(I_origin) # may change I_origin, use I_bgr after this line
    #cape_util.display(I_bgr)
    faces_xywh = vj_face.face_detect(I_aindane)
    # cape_util.display(I_face_rectangle)
    faces = [ I_bgr[y:y+h, x:x+w] for (x,y,w,h) in faces_xywh ]
    # for face in faces:
    #     cape_util.display( face )

    skins = [ apa_skin.skin_detect(face) for face in faces ]
    for skin in skins:
        cape_util.display( skin )
    return 0

if __name__ == '__main__':
    main()