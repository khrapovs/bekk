from cython_gsl cimport *
cimport cython
import numpy as np
cimport numpy as cnp

__all__ = ['likelihood']

cdef extern from 'math.h':
    double log(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def likelihood(double[:, :, :] hvar, double[:, :] innov):

    cdef:
        Py_ssize_t nstocks, nobs
        double[:] temp
        double[:, :] hvartemp
        double value = 0.0
        double logf = 0.0
        gsl_matrix_view Hvartemp
        gsl_vector_view Innov, Temp, Diag

    nobs, nstocks = hvar.shape[0], hvar.shape[1]

    temp = np.zeros(nstocks)
    Temp = gsl_vector_view_array(&temp[0], nstocks)

    for t in range(nobs):

        hvartemp = hvar[t].copy()
        Hvartemp = gsl_matrix_view_array(&hvartemp[0, 0], nstocks, nstocks)
        Innov = gsl_vector_view_array(&innov[t, 0], nstocks)

        gsl_linalg_cholesky_decomp(&Hvartemp.matrix)
        gsl_linalg_cholesky_solve(&Hvartemp.matrix, &Innov.vector, &Temp.vector)
        gsl_blas_ddot(&Innov.vector, &Temp.vector, &value)

        Diag = gsl_matrix_diagonal(&Hvartemp.matrix)
        gsl_vector_mul(&Diag.vector, &Diag.vector)

        for i in range(nstocks):
            value += log(gsl_vector_get(&Diag.vector, i))

        logf += value

    return logf
