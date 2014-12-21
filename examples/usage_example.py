#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import numpy as np
import scipy.linalg as sl

from bekk import BEKK, BEKKParams, simulate_bekk, regenerate_data


def test_bekk(nstocks=2, nobs=500, restriction='scalar', var_target=True,
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
    # Data file
    innov_file = '../data/innovations.npy'

    if simulate:
        # Simulate data
        innov = simulate_bekk(param_true, nobs=nobs, distr='student')
        np.savetxt(innov_file[:-4] + '.csv', innov, delimiter=",")

    else:
        # Regenerate real data
        regenerate_data(innov_file=innov_file, nstocks=nstocks, nobs=nobs)
        # Load data from the drive
        innov = np.load(innov_file)

    #innov = innov[np.abs(innov) < 2*innov.std(), np.newaxis]
    # Initialize the object
    bekk = BEKK(innov)
    # Estimate parameters
    bekk.estimate(param_start=param_true, param_true=param_true,
                  restriction=restriction, var_target=var_target,
                  method='Powell', log_file=log_file, parallel=False)
    bekk.print_error()


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    nstocks = 1
    var_target = True
    nobs = 500
    test_bekk(nstocks=nstocks, simulate=True, var_target=var_target,
              nobs=nobs, log_file='../logs/log_sim.txt')
#    test_bekk(nstocks=nstocks, simulate=False, var_target=var_target,
#              nobs=nobs, log_file='log_real.txt')
