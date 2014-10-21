#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import scipy.linalg as sl
import matplotlib.pylab as plt

from MGARCH.bekk import BEKK, simulate_bekk, init_parameters
from MGARCH.bekk import convert_abc_to_theta
from MGARCH.bekk import BEKKParameters

def test_simulate(nstocks=2, nobs=500):
    """Simulate and estimate BEKK model.

    Parameters
    ----------
    nstocks : int
        Number of stocks in the model
    nobs : int
        The length of time series

    """
    log_file = 'bekk_log.txt'
    with open(log_file, 'w') as texfile:
        texfile.write('')

    # scalar, diagonal, full
    restriction = 'scalar'
    # Variance targetign flag
    var_target = False
    # A, B, C - n x n matrices
    A = np.eye(nstocks) * .15
    B = np.eye(nstocks) * .95
    Craw = np.ones((nstocks, nstocks))*.5 + np.eye(nstocks)*.5
    C = sl.cholesky(Craw, 1)
    theta_true = convert_abc_to_theta(A, B, C, restriction, False)

    true_param = BEKKParameters(a_mat=A, b_mat=B, c_mat=C)

    # Simulate data
    innov = simulate_bekk(true_param, nobs=nobs)
    # Plot data
    plt.plot(innov)
    plt.show()

    # Initialize the object
    bekk = BEKK(innov)
    # Estimate parameters
    bekk.estimate(theta_start=theta_true, theta_true=theta_true,
                  restriction=restriction, var_target=var_target,
                  method='Powell', log_file=log_file)

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
    ret.plot()
    plt.show()

    # Create array of innovations
    innov = np.array(ret)
    np.save(innov_file, innov)
    np.savetxt(innov_file[:-4] + '.csv', innov, delimiter=",")

def test_real(method, theta_start, restriction, stage):
    # Load data
    innov_file = 'innovations.npy'
    # Load data from the drive
    innov = np.load(innov_file)

    #import numpy as np
    log_file = 'bekk_log_' + method + '_' + str(stage) + '.txt'
    # Clean log file
    with open(log_file, 'w') as texfile:
        texfile.write('')
    # Initialize the object
    bekk = BEKK(innov)
    # Restriction of the model, 'scalar', 'diagonal', 'full'
    bekk.restriction = restriction
    # Print results for each iteration?
    bekk.use_callback = False
    # Set log file name
    bekk.log_file = log_file
    # Do we want to download data and overwrite it on the drive?
    #regenerate_data(innov_file)

    # 'Newton-CG', 'dogleg', 'trust-ncg' require gradient
    #    methods = ['Nelder-Mead','Powell','CG','BFGS','L-BFGS-B',
    #               'TNC','COBYLA','SLSQP']
    # Optimization method
    bekk.method = method
    # Estimate parameters
    bekk.estimate(theta_start)
    # bekk.estimate(bekk.theta_final)
    return bekk

def simple_test():
    """Estimate BEKK.
    """
    # Load data
    innov_file = 'innovations.npy'
    # Load data from the drive
    innov = np.load(innov_file)
    nstocks = innov.shape[1]
    # Choose the model restriction: scalar, diagonal, full
    restriction = 'scalar'
    # Initialize parameters
    theta_start = init_parameters(restriction, nstocks)

    # Run first stage estimation
    test_real('L-BFGS-B', theta_start, restriction, 1)

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    test_simulate()