# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 21:14:41 2014

@author: khrapov
"""

import numpy as np
import scipy.linalg as scl
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from IPython import get_ipython

np.set_printoptions(precision = 2, suppress = True)
ipython = get_ipython()

N, K = 100, 9

A, C = [], []
for n in range(N):
    x = np.random.normal(size = (K, K))
    A.append(x.dot(x.T))

def loop_invert(A):
    B = []
    for n in range(N):
        B.append(np.linalg.inv(A[n]))
    return

def loop_invert2(A):
    B = []
    for n in range(N):
        B.append(scl.inv(A[n]))
    return

def sparse_invert(A):
    C = []
    for n in range(N):
        C.append(sp.coo_matrix(A[n]))

    C = sp.block_diag(C).tocsc()
    Cinv = spl.inv(C)
    return