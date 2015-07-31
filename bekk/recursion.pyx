from cython_gsl cimport *
cimport cython
import numpy as np
cimport numpy as cnp

__all__ = ['filter_var']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filter_var(double[:, :, :] hvar, double[:, :] innov,
               double[:, :] amat, double[:, :] bmat, double[:, :] cmat):

    cdef:
        Py_ssize_t nstocks, nobs
        double[:, :] intercept, temp, innov2
        gsl_matrix_view A, B, C, Intercept, Innov, Innov2, Hvar, Temp

    nobs, nstocks = hvar.shape[0], hvar.shape[1]
    intercept = np.zeros_like(cmat, float)
    temp = np.zeros_like(cmat, float)
    innov2 = np.zeros_like(cmat, float)

    A = gsl_matrix_view_array(&amat[0, 0], nstocks, nstocks)
    B = gsl_matrix_view_array(&bmat[0, 0], nstocks, nstocks)
    C = gsl_matrix_view_array(&cmat[0, 0], nstocks, nstocks)
    Intercept = gsl_matrix_view_array(&intercept[0, 0], nstocks, nstocks)
    Temp = gsl_matrix_view_array(&temp[0, 0], nstocks, nstocks)
    Innov2 = gsl_matrix_view_array(&innov2[0, 0], nstocks, nstocks)

    # cmat.dot(cmat.T)
    gsl_blas_dsyrk(CblasUpper, CblasNoTrans,
                   1.0, &C.matrix, 0.0, &Intercept.matrix)

    for t in range(1, nobs):

        hvar[t] = intercept.copy()

        Innov = gsl_matrix_view_array(&innov[t-1, 0], nstocks, 1)
        Hvar = gsl_matrix_view_array(&hvar[t, 0, 0], nstocks, nstocks)
        Hvar_lag = gsl_matrix_view_array(&hvar[t-1, 0, 0], nstocks, nstocks)

        # innov[t].dot(innov[t].T)
        gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                       1.0, &Innov.matrix, &Innov.matrix, 0.0, &Innov2.matrix)

        # a_mat.dot(innov2[i-1]).dot(a_mat.T)
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                       1.0, &A.matrix, &Innov2.matrix, 0.0, &Temp.matrix)
        gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                       1.0, &Temp.matrix, &A.matrix, 1.0, &Hvar.matrix)

        # b_mat.dot(hvar[i-1]).dot(b_mat.T)
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                       1.0, &B.matrix, &Hvar_lag.matrix, 0.0, &Temp.matrix)
        gsl_blas_dgemm(CblasNoTrans, CblasTrans,
                       1.0, &Temp.matrix, &B.matrix, 1.0, &Hvar.matrix)

    return np.asarray(hvar)
