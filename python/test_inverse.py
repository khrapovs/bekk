# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 21:14:41 2014

@author: khrapov
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from IPython import get_ipython

np.set_printoptions(precision = 2, suppress = True)
ipython = get_ipython()

N, K = 1000, 9

#%%
A, C = [], []
for n in range(N):
    x = np.random.normal(size = (K, K))
    A.append(x.dot(x.T))
    C.append(sp.coo_matrix(A[n]))

#%%
ipython.magic('%%time')

B = []
for n in range(N):
    B.append(np.linalg.inv(A[n]))

#%%
ipython.magic('%%time')

C = sp.block_diag(C).tocsc()
Cinv = spl.inv(C)

#%%