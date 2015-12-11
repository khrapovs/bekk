#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data generation
===============

"""

import numpy as np
import scipy.linalg as sl
import pandas as pd

from pandas_datareader import data
from skewstudent import SkewStudent

__all__ = ['simulate_bekk', 'download_data']


def simulate_bekk(param, nobs=1000, distr='normal', degf=10, lam=0):
    """Simulate data.

    Parameters
    ----------
    param : BEKKParams instance
        Attributes of this class hold parameter matrices
    nobs : int
        Number of observations to generate. Time series length
    distr : str
        Name of the distribution from which to generate innovations.

        Must be
            - 'normal'
            - 'student'
            - 'skewt'
    degf : int
        Degrees of freedom for Student or SkewStudent distributions
    lam : float
        Skewness parameter for Student or SkewStudent distributions.
        Must be between (-1, 1)

    Returns
    -------
    innov : (nobs, nstocks) array
        Multivariate innovation matrix

    """
    nstocks = param.amat.shape[0]
    if distr == 'normal':
        # Normal innovations
        mean, cov = np.zeros(nstocks), np.eye(nstocks)
        error = np.random.multivariate_normal(mean, cov, nobs)
    elif distr == 'student':
        # Student innovations
        error = np.random.standard_t(degf, size=(nobs, nstocks))
    elif distr == 'skewt':
        # Skewed Student innovations
        error = SkewStudent(eta=degf, lam=lam).rvs(size=(nobs, nstocks))
    else:
        raise ValueError('Unknown distribution!')
    # Standardize innovations
    error = (error - error.mean(0)) / error.std(0)
    hvar = np.empty((nobs, nstocks, nstocks))
    innov = np.zeros((nobs, nstocks))

    hvar[0] = param.get_uvar()
    intercept = param.cmat.dot(param.cmat.T)

    for i in range(1, nobs):
        innov2 = innov[i-1, np.newaxis].T * innov[i-1]
        hvar[i] = intercept + param.amat.dot(innov2).dot(param.amat.T) \
            + param.bmat.dot(hvar[i-1]).dot(param.bmat.T)
        hvar12 = sl.cholesky(hvar[i], 1)
        innov[i] = hvar12.dot(np.atleast_2d(error[i]).T).flatten()

    return innov, hvar


def download_data(tname='innovations', tickers=None, nobs=None):
    """Download stock market data and save it to disk.

    Parameters
    ----------
    innov_file : str
        Name of the file to save to
    nstocks : int
        Number of stocks to analyze
    nobs : int
        Number of observations in the time series

    Returns
    -------
    ret : DataFrame
        Demeaned returns

    """
    prices = []
    start, end = '2002-01-01', '2015-12-31'
    colname = 'Adj Close'
    for tic in tickers:
        stock = data.DataReader(tic, 'yahoo', start, end)[colname]
        stock.name = tic
        prices.append(stock)

    prices = pd.concat(prices, axis=1, join='inner')

    ret = (np.log(prices) - np.log(prices.shift(1))) * 100
    ret.dropna(inplace=True)
    ret = ret.apply(lambda x: x - x.mean())
    ret = ret.iloc[-nobs:] if nobs is not None else ret

    return ret
