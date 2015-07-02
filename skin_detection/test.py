import appendixa_skin_detect as apa
import copy_apa as cp

import cv2
import ipdb

IMG_DIR = '../resources/images/'

img = cv2.imread(IMG_DIR+'teaser_face.png')
ipdb.set_trace()
skin, mask = apa.skin_detect(img)
skincp, maskcp = cp.skin_detect(img)

print mask == maskcp, (mask==maskcp).all()