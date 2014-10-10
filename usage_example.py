#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pylab as plt

from MGARCH.bekk import BEKK, simulate_BEKK, estimate_H0, init_parameters
from MGARCH.bekk import convert_abc_to_theta

def test_simulate(n=2, T=100):
    log_file = 'bekk_log.txt'
    with open(log_file, 'w') as texfile:
        texfile.write('')
        
    restriction = 'scalar'
    # A, B, C - n x n matrices
    A = np.eye(n) * .25
    B = np.eye(n) * .95
    C = sp.linalg.cholesky(np.ones((n,n))*.5 + np.eye(n)*.5, 1)
    theta = convert_abc_to_theta(A, B, C, restriction, False)
    
    # Simulate data    
    u = simulate_BEKK(A, B, C, T=T)
    # Estimate stationary variance
    stationary_var = estimate_H0(u)
    # Compute the constant term
    CC = stationary_var - A.dot(stationary_var).dot(A.T) \
        - B.dot(stationary_var).dot(B.T)
    # Extract C parameter
    Cstart = sp.linalg.cholesky(CC, 1)
    
    # Initialize the object
    bekk = BEKK(u)
    
    var_target = True
    # Choose initial theta
    theta_start = convert_abc_to_theta(A, B, Cstart, restriction, var_target)
    
    # Estimate parameters
    bekk.estimate(theta_start=theta_start, theta_true=theta[:2*n**2],
                  var_target=var_target, method='Powell', log_file=log_file)
    
def regenerate_data(u_file):
    """Download and save data to disk.
    
    Parameters
    ----------
        u_file: str, name of the file to save to
    """
    import Quandl
    import pandas as pd
    
    token = open('Quandl.token', 'r').read()
    tickers = ["GOOG/NASDAQ_MSFT", "GOOG/NASDAQ_AAPL"]
    tickers2 = ["GOOG/NYSE_XOM", "GOOG/NYSE_OXY"]
    tickers3 = ["GOOG/NYSE_TGT", "GOOG/NYSE_WMT"]
    tickers.extend(tickers2)
    tickers.extend(tickers3)
    prices = []
    for tic in tickers:
        df = Quandl.get(tic, authtoken = token,
                        trim_start = "2000-01-01",
                        trim_end = "2007-12-31")[['Close']]
        df.rename(columns = {'Close' : tic}, inplace = True)
        prices.append(df)
    prices = pd.concat(prices, axis = 1)
    
    ret = (np.log(prices) - np.log(prices.shift(1))) * 100
    ret.dropna(inplace = True)
    ret.plot()
    plt.show()
    
    # Create array of innovations    
    u = np.array(ret.apply(lambda x: x - x.mean()))#[-2000:]
    print(u.shape)
    np.save(u_file, u)
    np.savetxt(u_file[:-4] + '.csv', u, delimiter = ",")

def test_real(method, theta_start, restriction, stage):
    # Load data    
    u_file = 'innovations.npy'
    # Load data from the drive
    u = np.load(u_file)
    
    #import numpy as np
    log_file = 'bekk_log_' + method + '_' + str(stage) + '.txt'
    # Clean log file
    with open(log_file, 'w') as texfile:
        texfile.write('')
    # Initialize the object
    bekk = BEKK(u)
    # Restriction of the model, 'scalar', 'diagonal', 'full'
    bekk.restriction = restriction
    # Print results for each iteration?
    bekk.use_callback = False
    # Set log file name
    bekk.log_file = log_file
    # Do we want to download data and overwrite it on the drive?
    #regenerate_data(u_file)
    
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
    u_file = 'innovations.npy'
    # Load data from the drive
    u = np.load(u_file)
    n = u.shape[1]
    # Choose the model restriction: scalar, diagonal, full
    restriction = 'scalar'
    # Initialize parameters
    theta_start = init_parameters(restriction, n)
    
    # Run first stage estimation
    test_real('L-BFGS-B', theta_start, restriction, 1)
 
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    test_simulate(n=2, T=500)