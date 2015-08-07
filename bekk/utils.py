# -*- coding: utf-8 -*-
"""
Helper functions

"""
from __future__ import print_function, division

import time
import contextlib
import itertools

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import scipy.linalg as scl

__all__ = ['estimate_uvar', 'plot_data', 'get_weight',
           'filter_var_python',  'likelihood_python']


def filter_var_python(hvar, innov, amat, bmat, cmat):
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
    innov2 = innov[:, np.newaxis, :] * innov[:, :, np.newaxis]
    intercept = cmat.dot(cmat.T)
    for i in range(1, nobs):
        hvar[i] = intercept + amat.dot(innov2[i-1]).dot(amat.T) \
            + bmat.dot(hvar[i-1]).dot(bmat.T)

    return hvar


def likelihood_python(hvar, innov):
    """Likelihood function.

    Parameters
    ----------
    innov : (nobs, nstocks) array
        inovations
    hvar : (nobs, nstocks, nstocks) array
        variance/covariances

    Returns
    -------
    fvalue : float
        log-likelihood contribution

    """
    lower = True
    fvalue = 0
    for innovi, hvari in zip(innov, hvar):
        hvari, lower = scl.cho_factor(hvari, lower=lower, check_finite=False)
        norm_innov = scl.cho_solve((hvari, lower), innovi, check_finite=False)
        fvalue += (np.log(np.diag(hvari)**2) + norm_innov * innovi).sum()

    return fvalue


def estimate_uvar(innov):
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


def format_time(t):
    """Format time for nice printing.

    Parameters
    ----------
    t : float
        Time in seconds

    Returns
    -------
    format template

    """
    if t > 60 or t == 0:
        units = 'min'
        t /= 60
    elif t > 1:
        units = 's'
    elif t > 1e-3:
        units = 'ms'
        t *= 1e3
    elif t > 1e-6:
        units = 'us'
        t *= 1e6
    else:
        units = 'ns'
        t *= 1e9
    return '%.1f %s' % (t, units)


@contextlib.contextmanager
def take_time(desc):
    """Context manager for timing the code.

    Parameters
    ----------
    desc : str
        Description of the code

    Example
    -------
    >>> with take_time('Estimation'):
    >>>    estimate()

    """
    t0 = time.time()
    yield
    dt = time.time() - t0
    print('%s took %s' % (desc, format_time(dt)))


def get_weight(groups=None, nitems=1):
    """Generate weighting matrices given groups.

    Parameters
    ----------
    groups : list of tuples
        Encoded groups of items
    nitems : int
        Total number of items

    Returns
    -------
    (ngroups, nitems, nitems) array
        Spatial weights

    """
    if groups is None:
        ncat = 1
    else:
        ncat = len(groups)

    weight = np.zeros((ncat, nitems, nitems))
    for i in range(ncat):
        for id1, id2 in itertools.product(groups[i], groups[i]):
            if id1 != id2:
                weight[i, id1, id2] = 1
        norm = weight[i].sum(0)[:, np.newaxis]
        norm[norm == 0] = 1
        weight[i] /= norm

    return weight
