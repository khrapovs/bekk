from cython_gsl cimport *
cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *

__all__ = ['likelihood_gauss', 'likelihood_gauss2']

cdef extern from 'math.h':
    double log(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def likelihood_gauss(double[:, :, :] hvar, double[:, :] innov):

    cdef:
        Py_ssize_t nstocks, nobs, t, i
        double[:] temp
        double[:, :] hvartemp
        double value = 0.0
        double[:] logf
        gsl_matrix_view Hvartemp
        gsl_vector_view Innov, Temp, Diag

    nobs, nstocks = hvar.shape[0], hvar.shape[1]

    temp = np.zeros(nstocks, float)
    logf = np.zeros(nobs, float)
    Temp = gsl_vector_view_array(&temp[0], nstocks)

    for t in range(nobs):

        hvartemp = hvar[t].copy()
        Hvartemp = gsl_matrix_view_array(&hvartemp[0, 0], nstocks, nstocks)
        Innov = gsl_vector_view_array(&innov[t, 0], nstocks)

        gsl_linalg_cholesky_decomp(&Hvartemp.matrix)
        gsl_linalg_cholesky_solve(&Hvartemp.matrix,
                                  &Innov.vector, &Temp.vector)
        gsl_blas_ddot(&Innov.vector, &Temp.vector, &value)

        Diag = gsl_matrix_diagonal(&Hvartemp.matrix)
        gsl_vector_mul(&Diag.vector, &Diag.vector)

        for i in range(nstocks):
            value += log(gsl_vector_get(&Diag.vector, i))

        logf[t] = value

    return np.asarray(logf).sum()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def likelihood_gauss2(double[:, :, :] hvar, double[:, :] innov):

    cdef:
        Py_ssize_t t, i
        int info = 0
        int nrhs = 1
        int inc = 1
        int nobs = hvar.shape[0]
        int n = hvar.shape[1]
        double[:] temp = np.zeros(n, float)
        double[:] logf = np.zeros(nobs, float)

    for t in range(nobs):

        temp = innov[t].copy()

        # H^(-1/2)
        # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
        dpotrf('U', &n, &hvar[t, 0, 0], &n, &info)

        # uH^(-1)
        # http://www.math.utah.edu/software/lapack/lapack-d/dpotrs.html
        dpotrs('U', &n, &nrhs, &hvar[t, 0, 0], &n, &temp[0], &n, &info)

        # uH^(-1)u'
        # http://www.mathkeisan.com/usersguide/man/ddot.html
        logf[t] = ddot(&n, &temp[0], &inc, &innov[t, 0], &inc)

        # log|H|
        for i in range(n):
            logf[t] += log(hvar[t, i, i] ** 2)

    return np.asarray(logf).sum()
