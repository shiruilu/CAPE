import cv2
import matplotlib.pyplot as plt
from kde_for_scipy.core.kde import kde
from scipy.stats import gaussian_kde


class SmoothedHist:
    IMG_DIR = '../resources/images/'

    @staticmethod
    def plot_smoothed_hist(data, N=None, MIN=None, MAX=None):
        """
        Keyword Arguments:
        data -- 1d
        N    -- number of bins
        MIN  -- min value, default is min(data)
        MAX  -- max value, default is max(data)
        """
        bw, mesh, kdense = kde(data, N, MIN, MAX)
        print bw, mesh, kdense
        plt.plot(mesh, kdense)
        plt.title('#bins=' + str(N))
        plt.show()
        SmoothedHist.test_scipy_kde(data, bw, mesh)

    @staticmethod
    def test_scipy_kde(data, bw, mesh):
        """ use kde_for_scipy to aid mesh generation """
        kde_sci = gaussian_kde(data)
        plt.plot(mesh, kde_sci.evaluate(mesh))
        plt.title('scipy.stats')
        plt.show()

    @staticmethod
    def main():
        IMG_NAME = 'teaser_face'
        imgLab = cv2.cvtColor(
            cv2.imread(SmoothedHist.IMG_DIR + IMG_NAME + '.png'),
            cv2.COLOR_BGR2LAB)
        imgLab = imgLab.astype('uint8')
        SmoothedHist.plot_smoothed_hist(imgLab[..., 0].ravel())
        # save image data for R silverman analysis
        # import os
        # import numpy as np
        # DATA_DIR = os.path.expanduser('~/Dropbox/dropcode/r/')
        # np.savetxt(DATA_DIR+IMG_NAME+'.csv', imgLab[...,0].ravel(), delimiter=',')


####################################################

if __name__ == '__main__':
    SmoothedHist.main()
