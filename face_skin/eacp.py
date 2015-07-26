"""
Created on June 30, 2015
@author: shiruilu

Edge-aware constraint propagation
From "Interactive Local Adjustment of Tonal Values"[LFUS06]
"""
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def EACP(G, I, W=None, lambda_=0.2, alpha=1.0, eps=1e-4):
    """
    Edge-aware constraint propagation
    From "Interactive Local Adjustment of Tonal Values"[LFUS06]
    ARGs:
    -----
    G(A): will be g(x) in 3.2 of LFUS06, desired result.
    I: will be transformed to L (log luminance channel)
    W: float,(0-1) will be flattened to w, specifies a weight for each constrained pixel
    """
    if G.shape != I.shape:
        raise ValueError('A and I are not in the same size')
    if W == None:
        W = np.ones(G.shape)
    L = np.log(I+eps) # avoid log of 0
    # L = I
    g = G.flatten(1)
    w = W.flatten(1)
    s = L.shape

    k = np.prod(s)
    # L_i - L_j along y axis
    dy = np.diff(L, 1, 0)
    dy = -lambda_ / (np.absolute(dy) ** alpha + eps)
    dy = np.vstack((dy, np.zeros(s[1], )))
    dy = dy.flatten(1)
    # L_i - L_j along x axis
    dx = np.diff(L, 1, 1)
    dx = -lambda_ / (np.absolute(dx) ** alpha + eps)
    dx = np.hstack((dx, np.zeros(s[0], )[:, np.newaxis]))
    dx = dx.flatten(1)
    # A case: j \in N_4(i)  (neighbors of diagonal line)
    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)
    # A case: i=j   (diagonal line)
    d = w - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k) # A: put together
    f = spsolve(a, w*g).reshape(s[::-1]) # slove Af  =  b =w*g and restore 2d
    A = np.rollaxis(f,1)
    # A = np.clip( _out*255.0, 0, 255).astype('uint8')
    return A