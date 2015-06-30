import os
import sys

source_dirs = ['cape_util', 'aindane', 'face_detection', 'skin_detection', 'wls_filter']

for d in source_dirs:
    sys.path.insert( 0, os.path.join(os.getcwd(), '../'+d) )

import appendixa_skin_detect as apa_skin

import numpy as np
import cv2
from matplotlib import pyplot as plt

IMG_DIR = '../resources/images/'

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

def sufficient_skin(skin_img, portion):
    '''
    To remove false examples of detected face

    skin_img: (masked) non-skin area filled with 0 (black),
              skin area copied from origin
    protion:  % of skin area in image
    '''
    img_gray = cv2.cvtColor( skin_img, cv2.COLOR_BGR2GRAY )
    ## ? bitwise_and behavior: if mask !=0, fill with 1
    print 'skin portion: ' ,1.0*np.sum( img_gray !=1 ) / (img_gray.shape[0] * img_gray.shape[1])
    return np.sum(img_gray !=1) >= portion *skin_img.shape[0] *skin_img.shape[1]

def face_detect(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_xywh = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # remove false positives in faces_xywh(half face) if skin portion < 0.3
    faces_xywh = [(x,y,w,h) for (x,y,w,h) in faces_xywh  \
                  if sufficient_skin(apa_skin.skin_detect(img[y:y+h, x:x+w])[0], 0.41)]
    '''
    for (x,y,w,h) in faces_xywh:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    '''
    return faces_xywh

def display(img):
    plt.imshow(img)
    plt.title('face_detection')
    plt.show()

def _test_face_detect():
    img = cv2.imread(IMG_DIR+'input_teaser.png')
    # CAUTION: will change the color of face rectangle
    display( face_detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ) )
    return 0

if __name__ == '__main__':
    _test_face_detect()