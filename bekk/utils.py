# -*- coding: utf-8 -*-
"""
Helper functions

"""

from __future__ import print_function, division

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import scipy.linalg as scl
from functools import reduce
import multiprocessing as mp
import numba as nb
import scipy.sparse as scs

__all__ = ['_bekk_recursion', '_product_cc',
           '_product_aba', '_filter_var', '_contribution',
           'estimate_h0', 'plot_data']


def _bekk_recursion(param, hzero, hone, htwo):
    """BEKK recursion.

    Parameters
    ----------
    param : BEKKParams instance
        Model parameters
    hzero : (nstocks, nstocks) array
        Initial matrix
    hone : (nstocks, nstocks) array
        Squared innovations matrix
    htwo : (nstocks, nstocks) array
        Old matrix

    Returns
    -------
    hnew : (nstocks, nstocks) array
        Updated variance matrix

    """
    return hzero + _product_aba(param.a_mat, hone) \
        + _product_aba(param.b_mat, htwo)


def _product_cc(mat):
    """Compute CC'.

    Parameters
    ----------
    mat : 2dim square array

    Returns
    -------
    mat : 2dim square array

    """
    return mat.dot(mat.T)


def _product_aba(a_mat, b_mat):
    """Compute ABA'.

    Parameters
    ----------
    a_mat, b_mat : 2dim arrays

    Returns
    -------
    mat : 2dim square array

    """
    return reduce(np.dot, [a_mat, b_mat, a_mat.T])

#@nb.jit("float32[:,:,:](float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])")
@nb.autojit
def _filter_var(innov, c_mat, a_mat, b_mat, uvar):
    """Filter out variances and covariances of innovations.

    Parameters
    ----------
    innov : (nobs, nstocks) array
        Return innovations
    param : instance of BEKKParams class
        Attributes of this class hold parameter matrices

    Returns
    -------
    hvar : (nobs, nstocks, nstocks) array
        Variances and covariances of innovations

    """
    nobs, nstocks = innov.shape
    hvar = np.empty((nobs, nstocks, nstocks))
    hvar[0] = uvar
    cc_mat = c_mat.dot(c_mat.T)

    for i in range(1, nobs):
        innov2 = innov[i-1, np.newaxis].T * innov[i-1]
        hvar[i] = cc_mat + a_mat.dot(innov2).dot(a_mat.T) \
            + b_mat.dot(hvar[i-1]).dot(b_mat.T)

    return hvar

@nb.autojit
def _filter_var2(hvar, innov, c_mat, a_mat, b_mat):
    """Filter out variances and covariances of innovations.

    Parameters
    ----------
    innov : (nobs, nstocks) array
        Return innovations
    param : instance of BEKKParams class
        Attributes of this class hold parameter matrices

    Returns
    -------
    hvar : (nobs, nstocks, nstocks) array
        Variances and covariances of innovations

    """
    nobs, nstocks = innov.shape
    cc_mat = c_mat.dot(c_mat.T)

    for i in range(1, nobs):
        innov2 = innov[i-1, np.newaxis].T * innov[i-1]
        hvar[i*nstocks:(i+1)*nstocks, i*nstocks:(i+1)*nstocks] \
            = cc_mat + a_mat.dot(innov2).dot(a_mat.T) \
            + b_mat.dot(hvar[(i-1)*nstocks:i*nstocks, (i-1)*nstocks:i*nstocks].dot(b_mat.T))

    return hvar

@nb.autojit
def likelihood(hvar, innov):
    """Likelihood function.

    Parameters
    ----------
    innov : (nstocks,) array
        inovations
    hvar : (nstocks, nstocks) array
        variance/covariances
    parallel : bool
        Whether to use multiprocessing

    Returns
    -------
    fvalue : float
        log-likelihood contribution
    bad : bool
        True if something is wrong

    """
    sumf = 0
    for innovi, hvari in zip(innov, hvar):
        lower = True
        scl.cho_factor(hvari, lower=lower, overwrite_a=True, check_finite=False)
        norm_innov = scl.cho_solve((hvari, lower), innovi, check_finite=False)
        sumf += (2 * np.log(np.diag(hvari)) + norm_innov * innovi).sum()
    return sumf


def likelihood2(hvar, innov):
    """Likelihood function.

    Parameters
    ----------
    innov : (nstocks,) array
        inovations
    hvar : (nstocks, nstocks) array
        variance/covariances
    parallel : bool
        Whether to use multiprocessing

    Returns
    -------
    fvalue : float
        log-likelihood contribution
    bad : bool
        True if something is wrong

    """
    factor = scs.linalg.splu(hvar)
    diag_factor = np.diag(factor.U.toarray())
    innov = innov.flatten()
    norm_innov = factor.solve(innov)
    sumf = np.log(diag_factor[~np.isnan(diag_factor)]**2).sum() \
        + (norm_innov * innov).sum()
    return sumf

@nb.autojit
def _contribution(innov, hvar):
    """Contribution to the log-likelihood function for each observation.

    Parameters
    ----------
    innov: (nstocks,) array
        inovations
    hvar: (nstocks, nstocks) array
        variance/covariances

    Returns
    -------
    fvalue : float
        log-likelihood contribution
    bad : bool
        True if something is wrong

    """

    lower = True
    scl.cho_factor(hvar, lower=lower, overwrite_a=True, check_finite=False)
    norm_innov = scl.cho_solve((hvar, lower), innov, check_finite=False)
    fvalue = (2 * np.log(np.diag(hvar)) + norm_innov * innov).sum()

    return fvalue

def _contribution_good(innov, hvar):
    """Contribution to the log-likelihood function for each observation.

    Parameters
    ----------
    innov: (nstocks,) array
        inovations
    hvar: (nstocks, nstocks) array
        variance/covariances

    Returns
    -------
    fvalue : float
        log-likelihood contribution
    bad : bool
        True if something is wrong

    """

    lower = True
    try:
        scl.cho_factor(hvar, lower=lower, overwrite_a=True, check_finite=False)
    except (scl.LinAlgError, ValueError):
        return 1e10, True

    norm_innov = scl.cho_solve((hvar, lower), innov, check_finite=False)
    fvalue = (2 * np.log(np.diag(hvar)) + norm_innov * innov).sum()

    if np.isinf(fvalue):
        return 1e10, True
    else:
        return fvalue, False

def estimate_h0(innov):
    """Estimate unconditional realized covariance matrix.

    Parameters
    ----------
    innov: (nobs, nstocks) array
        inovations

    Returns
    -------
    (nstocks, nstocks) array
        E[innov' * innov]

    """
    return innov.T.dot(innov) / innov.shape[0]


def plot_data(innov, hvar):
    """Plot time series of hvar and u elements.

    Parameters
    ----------
    innov: (nobs, nstocks) array
        innovations
    hvar: (nobs, nstocks, nstocks) array
        variance/covariances

    """
    sns.set_context('paper')
    nobs, nstocks = innov.shape
    axes = plt.subplots(nrows=nstocks**2, ncols=1)[1]
    for axi, i in zip(axes, range(nstocks**2)):
        axi.plot(range(nobs), hvar.reshape([nobs, nstocks**2])[:, i])
    plt.plot()

    axes = plt.subplots(nrows=nstocks, ncols=1)[1]
    for axi, i in zip(axes, range(nstocks)):
        axi.plot(range(nobs), innov[:, i])
    plt.plot()

def find_stationary_var(hvar, param):
    """Find fixed point of H = CC' + AHA' + BHB' given A, B, C.

    Parameters
    ----------
    innov: (nobs, nstocks) array
        innovations
    hvar: (nstocks, nstocks) array
        variance/covariances

    Returns
    -------
    hvarnew : (nstocks, nstocks) array
        Stationary variance amtrix

    """
    return _bekk_recursion(param, _product_cc(param.c_mat), hvar, hvar)
