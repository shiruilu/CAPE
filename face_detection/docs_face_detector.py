import numpy as np
import cv2
from matplotlib import pyplot as plt

IMG_DIR = '../resources/images/'

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

def face_detect(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),
    '''
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    '''
    return faces#, img

def display(img):
    plt.imshow(img)
    plt.title('face_detection')
    plt.show()

def _test_face_detect():
    img = cv2.imread(IMG_DIR+'input_teaser.png')
    display( face_detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ) ) # CAUTION: will change the color of face rectangle
    return 0

if __name__ == '__main__':
    _test_face_detect()