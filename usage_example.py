#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import scipy.linalg as sl

from MGARCH.bekk import BEKK, simulate_bekk
from MGARCH.bekk import BEKKParams


def test_simulate(nstocks=2, nobs=500,
                  restriction='scalar', var_target=True,
                  simulate=True, log_file='bekk_log.txt'):
    """Simulate and estimate BEKK model.

    Parameters
    ----------
    nstocks : int
        Number of stocks in the model
    nobs : int
        The length of time series
    restriction : str
        Restriction on model parameters. Can be
            - 'full'
            - 'diagonal'
            - 'scalar'
    var_target : bool
        Variance targeting flag.
        If True, then unconditonal variance is estimated on the first step.
        The rest of parameters are estimated on the second step.
    simulate : bool
        Whether to simulate the data (True) or load actual returns (False).
    log_file : str
        Name of the log file to output results.

    """
    with open(log_file, 'w') as texfile:
        texfile.write('')

    # A, B, C - n x n matrices
    A = np.eye(nstocks) * .3
    B = np.eye(nstocks) * .92
    Craw = np.ones((nstocks, nstocks))*.5 + np.eye(nstocks)*.5
    C = sl.cholesky(Craw, 1)

    param_true = BEKKParams(a_mat=A, b_mat=B, c_mat=C,
                            restriction=restriction, var_target=var_target)

    if simulate:
        # Simulate data
        innov = simulate_bekk(param_true, nobs=nobs)

    else:
        # Data file
        innov_file = 'innovations.npy'
        # Regenerate real data
        regenerate_data(innov_file=innov_file, nstocks=nstocks, nobs=nobs)
        # Load data from the drive
        innov = np.load(innov_file)

    # Initialize the object
    bekk = BEKK(innov)
    # Estimate parameters
    bekk.estimate(param_start=param_true, param_true=param_true,
                  restriction=restriction, var_target=var_target,
                  method='Powell', log_file=log_file, parallel=False)


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


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    nstocks = 1
    var_target = False
    test_simulate(nstocks=nstocks, simulate=True, var_target=var_target,
                  log_file='log_sim.txt')
    test_simulate(nstocks=nstocks, simulate=False, var_target=var_target,
                  log_file='log_real.txt')