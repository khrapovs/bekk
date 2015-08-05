cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *

__all__ = ['likelihood_gauss']

cdef extern from 'math.h':
    double log(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def likelihood_gauss(double[:, :, :] hvar, double[:, :] innov):

    cdef:
        Py_ssize_t t, i
        int info = 0
        int nrhs = 1
        int inc = 1
        int nobs = hvar.shape[0]
        int n = hvar.shape[1]
        double[:] temp = np.zeros(n, float)
        double[:] logf = np.zeros(nobs, float)
        double[:, :] hvarcopy = np.zeros((n, n), float)

    for t in range(nobs):

        temp = innov[t].copy()
        hvarcopy = hvar[t].copy()

        # H^(-1/2)
        # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
        dpotrf('U', &n, &hvarcopy[0, 0], &n, &info)

        # uH^(-1)
        # http://www.math.utah.edu/software/lapack/lapack-d/dpotrs.html
        dpotrs('U', &n, &nrhs, &hvarcopy[0, 0], &n, &temp[0], &n, &info)

        # uH^(-1)u'
        # http://www.mathkeisan.com/usersguide/man/ddot.html
        logf[t] = ddot(&n, &temp[0], &inc, &innov[t, 0], &inc)

        # log|H|
        for i in range(n):
            logf[t] += log(hvarcopy[i, i] ** 2)

    return np.asarray(logf).sum()
