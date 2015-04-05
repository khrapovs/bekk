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

__all__ = ['_bekk_recursion', '_product_cc',
           '_product_aba', 'filter_var', '_contribution',
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


def filter_var(innov, param):
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
    hvar[0] = param.unconditional_var()
    cc_mat = _product_cc(param.c_mat)

    for i in range(1, nobs):
        innov2 = innov[i-1, np.newaxis].T * innov[i-1]
        hvar[i] = _bekk_recursion(param, cc_mat, innov2, hvar[i-1])

    return hvar


def likelihood(hvar, innov, parallel):
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
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(_contribution, zip(innov, hvar))
        values, bad = zip(*results)
        sumf = np.array(values).sum()
        bad = np.array(bad).any()
    else:
        sumf = 0
        for innovi, hvari in zip(innov, hvar):
            fvalue, bad = _contribution(innovi, hvari)
            if bad:
                break
            sumf += fvalue
    return sumf, bad


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
