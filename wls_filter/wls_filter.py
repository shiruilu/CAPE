import cv2
import numpy
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

IMG_DIR = '../resources/images/'
small_eps = 0.0001

def wlsfilter(image, lambda_, alpha):
    s = image.shape

    k = numpy.prod(s)
    #image = original_image.astype(numpy.float)

    dy = numpy.diff(image, 1, 0)
    dy = -lambda_ / (numpy.absolute(dy) ** alpha + small_eps)
    dy = numpy.vstack((dy, numpy.zeros(s[1], )))
    dy = dy.flatten(1)

    dx = numpy.diff(image, 1, 1)
    dx = -lambda_ / (numpy.absolute(dx) ** alpha + small_eps)
    dx = numpy.hstack((dx, numpy.zeros(s[0], )[:, numpy.newaxis]))
    dx = dx.flatten(1)

    a = spdiags(numpy.vstack((dx, dy)), [-s[0], -1], k, k)

    d = 1 - (dx + numpy.roll(dx, s[0]) + dy + numpy.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k)
    out = spsolve(a, image.flatten(1)).reshape(s[::-1])
    return out, numpy.rollaxis(image,1)-out

def test_wlsfilter():
    lambda_ = 0.1
    alpha = 1.2
    image = cv2.imread(IMG_DIR+'easter.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(numpy.float)/255.0 #don't forget to normalize
    out, detail = wlsfilter(image, lambda_, alpha)
    plt.imshow(numpy.hstack([image,numpy.rollaxis(out,1),numpy.rollaxis(detail,1)]), 'gray')
    plt.show()

if __name__ == '__main__':
    test_wlsfilter()