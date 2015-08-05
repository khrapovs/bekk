cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *

__all__ = ['filter_var']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filter_var(double[:, :, :] hvar, double[:, :] innov,
               double[:, :] amat, double[:, :] bmat, double[:, :] cmat):

    cdef:
        int inc = 1
        int nobs = innov.shape[0]
        int n = innov.shape[1]
        double alpha = 1.0
        double beta = 0.0
        double beta2 = 1.0
        double scale = 1.0 / nobs
        double[:] temp = np.empty(n, float)
        double[:, :] temp2 = np.empty((n, n), float)
        double[:, :] intercept = np.empty((n, n), float)

    # CC'
    # http://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    dsyrk('U', 'N', &n, &n, &alpha, &cmat[0, 0], &n,
          &beta, &intercept[0, 0], &n)

    for t in range(1, nobs):

        hvar[t] = intercept.copy()

        # Au
        # http://www.math.utah.edu/software/lapack/lapack-blas/dgemv.html
        dgemv('N', &n, &n, &alpha, &amat[0, 0], &n, &innov[t-1, 0], &inc,
              &beta, &temp[0], &inc)

        # Auu'A'
        # http://www.mathkeisan.com/usersguide/man/ddot.html
        dsyr('U', &n, &alpha, &temp[0], &inc, &hvar[t, 0, 0], &n)

        # BH
        # http://www.math.utah.edu/software/lapack/lapack-blas/dsymm.html
        dsymm('R', 'U', &n, &n, &alpha, &hvar[t-1, 0, 0], &n,
              &bmat[0, 0], &n, &beta, &temp2[0, 0], &n)

        # BHB'
        # http://www.math.utah.edu/software/lapack/lapack-blas/dgemm.html
        dgemm('N', 'T', &n, &n, &n, &alpha, &temp2[0, 0], &n,
              &bmat[0, 0], &n, &beta2, &hvar[t, 0, 0], &n)

    return np.asarray(hvar)
