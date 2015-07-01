import cv2
import numpy
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

IMG_DIR = '../resources/images/'

def EACP_core(G, I, lambda_=0.2, alpha=1, small_eps=1e-4):
    """
    ? wls filter is using EACP as core idea
    """
    

def wlsfilter(image_orig, lambda_=0.1, alpha=1.2, small_eps=1e-4):
    """
    !!! Note: returning 0-255 may lose precision,
              or cause addition to more than 255(overflow)

    ARGs:
    -----
    image: 0-255, uint8, single channel (e.g. grayscale or single L)
    lambda_:
    alpha:

    RETURN:
    -----
    out: base, 0-255, uint8
    detail: detail, 0-255, uint8
    """
    image = image_orig.astype(numpy.float)/255.0
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
    _out = spsolve(a, image.flatten(1)).reshape(s[::-1])
    _out = numpy.rollaxis(_out,1)
    # out = numpy.clip( _out*255.0, 0, 255).astype('uint8')
    out = numpy.rint( _out*255.0 ).astype('uint8')
    # _detail = image - _out
    # detail = numpy.clip( _detail*255.0, 0, 255 ).astype('uint8')
    detail = image_orig - out
    return out, detail

def test_wlsfilter():
    """deprecated, need to remove rollaxis"""
    lambda_ = 0.1
    alpha = 1.2
    image = cv2.imread(IMG_DIR+'easter.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(numpy.float)/255.0 #don't forget to normalize
    out, detail = wlsfilter(image, lambda_, alpha)
    plt.imshow(numpy.hstack([image,numpy.rollaxis(out,1),numpy.rollaxis(detail,1)]), 'gray')
    plt.show()

def test_luminance():
    lambda_ = 0.1
    alpha = 1.2
    image = cv2.imread(IMG_DIR+'input_teaser.png')
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_lumi = image_LAB[...,0].astype(numpy.float)/255.0 #don't forget to normalize
    out, detail = wlsfilter(image_lumi, lambda_, alpha)
    image_base = numpy.zeros(image.shape)
    image_base[..., 0] = (numpy.rollaxis(out,1) *255.0)
    image_base[..., 1] = image_LAB[...,1]
    image_base[..., 2] = image_LAB[...,2]
    numpy.clip(image_base, 0, 255, out=image_base)
    image_base = image_base.astype('uint8')

    image_detail = numpy.zeros(image.shape)
    image_detail[..., 0] = (numpy.rollaxis(detail,1) *255.0)
    image_detail[..., 1] = image_LAB[...,1]
    image_detail[..., 2] = image_LAB[...,2]
    numpy.clip(image_detail, 0, 255, out=image_detail)
    image_detail = image_detail.astype('uint8')

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_base_RGB = cv2.cvtColor(image_base, cv2.COLOR_LAB2RGB)
    image_detail_RGB = cv2.cvtColor(image_detail, cv2.COLOR_LAB2RGB)
    plt.imshow(numpy.hstack([image_RGB, image_base_RGB, image_detail_RGB]), cmap='jet')
    plt.show()

if __name__ == '__main__':
    test_luminance()