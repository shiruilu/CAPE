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
        plt.plot(mesh, kdense); plt.title('#bins='+str(N)); plt.show()
        SmoothedHist.test_scipy_kde(data, bw, mesh)

    @staticmethod
    def test_scipy_kde(data, bw, mesh):
        """ use kde_for_scipy to aid bandwidth selection """
        kde_sci = gaussian_kde(data, bw)
        plt.plot(mesh, kde_sci.evaluate(mesh)); plt.title('scipy.stats'); plt.show()


    @staticmethod
    def main():
        imgLab = cv2.cvtColor( cv2.imread(SmoothedHist.IMG_DIR+'teaser_face.png'),
                               cv2.COLOR_BGR2LAB )
        imgLab = imgLab.astype('uint8')
        # for i in xrange(5,255,5):
        SmoothedHist.plot_smoothed_hist(imgLab[...,0].ravel())



####################################################

if __name__ == '__main__':
    SmoothedHist.main()
