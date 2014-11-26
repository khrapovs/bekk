#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate data.

"""

import numpy as np
import scipy.linalg as sl

from MGARCH import _bekk_recursion, _product_cc


def simulate_bekk(param, nobs=1000, distr='normal', degf=10):
    """Simulate data.

    Parameters
    ----------
    param : instance of BEKKParams class
        Attributes of this class hold parameter matrices
    nobs : int
        Number of observations to generate. Time series length
    distr : str
        Name of the distribution from which to generate innovations.
        Must be
            - normal
            - student

    Returns
    -------
    innov : (nobs, nstocks) array
        Multivariate innovation matrix

    """
    nstocks = param.a_mat.shape[0]
    if distr == 'normal':
        # Normal innovations
        mean, cov = np.zeros(nstocks), np.eye(nstocks)
        error = np.random.multivariate_normal(mean, cov, nobs)
    elif distr == 'student':
        # Student innovations
        error = np.random.standard_t(degf, size=(nobs, nstocks))
    else:
        raise ValueError('Unknown distribution!')
    # Standardize innovations
    error = (error - error.mean(0)) / error.std(0)
    hvar = np.empty((nobs, nstocks, nstocks))
    innov = np.zeros((nobs, nstocks))

    hvar[0] = param.unconditional_var()
    cc_mat = _product_cc(param.c_mat)

    for i in range(1, nobs):
        innov2 = innov[i-1, np.newaxis].T * innov[i-1]
        hvar[i] = _bekk_recursion(param, cc_mat, innov2, hvar[i-1])
        hvar12 = sl.cholesky(hvar[i], 1)
        innov[i] = hvar12.dot(np.atleast_2d(error[i]).T).flatten()

    from scipy.stats import kurtosis
    print('Kurtosis of the data: ', kurtosis(error))

    return innov

def regenerate_data(innov_file='innovations.npy', nstocks=2, nobs=None):
    """Download and save data to disk.

    Parameters
    ----------
    innov_file : str
        Name of the file to save to
    nstocks : int
        Number of stocks to analyze
    nobs : int
        Number of observations in the time series

    """
    import Quandl
    import pandas as pd

    token = open('Quandl.token', 'r').read()
    tickers = ["GOOG/NASDAQ_MSFT", "GOOG/NASDAQ_AAPL",
               "GOOG/NYSE_XOM", "GOOG/NYSE_OXY",
               "GOOG/NYSE_TGT", "GOOG/NYSE_WMT"]
    prices = []
    for tic in tickers[:nstocks]:
        df = Quandl.get(tic, authtoken=token,
                        trim_start="2001-01-01",
                        trim_end="2012-12-31")[['Close']]
        df.rename(columns={'Close' : tic}, inplace=True)
        prices.append(df)
    prices = pd.concat(prices, axis=1)

    ret = (np.log(prices) - np.log(prices.shift(1))) * 100
    ret.dropna(inplace = True)
    ret = ret.apply(lambda x: x - x.mean()).iloc[:nobs]

    # Create array of innovations
    innov = np.array(ret)
    np.save(innov_file, innov)
    np.savetxt(innov_file[:-4] + '.csv', innov, delimiter=",")
