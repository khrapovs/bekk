#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import time

import numpy as np
import scipy.linalg as sl

from bekk import BEKK, BEKKParams, simulate_bekk, regenerate_data, plot_data
from bekk import filter_var_python, filter_var_numba
from bekk import likelihood_python, likelihood_numba
from bekk import recursion, likelihood


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
    A = np.eye(nstocks) * .09**.5
    B = np.eye(nstocks) * .9**.5
    # Craw = np.ones((nstocks, nstocks))*.5 + np.eye(nstocks)*.5
    # Choose intercept to normalize unconditional variance to one
    Craw = np.eye(nstocks) - A.dot(A) - B.dot(B)
    C = sl.cholesky(Craw, 1)

    param_true = BEKKParams(a_mat=A, b_mat=B, c_mat=C,
                            restriction=restriction, var_target=var_target)
    # Data file
    innov_file = '../data/innovations.npy'

    if simulate:
        # Simulate data
        innov = simulate_bekk(param_true, nobs=nobs, distr='skewt', degf=30)
        np.savetxt(innov_file[:-4] + '.csv', innov, delimiter=",")

    else:
        # Regenerate real data
        regenerate_data(innov_file=innov_file, nstocks=nstocks, nobs=nobs)
        # Load data from the drive
        innov = np.load(innov_file)

    # Initialize the object
    bekk = BEKK(innov)
    # Estimate parameters
    for sparse in [True, False]:
        time_start = time.time()
        bekk.estimate(param_start=param_true, param_true=param_true,
                      restriction=restriction, var_target=var_target,
                      method='SLSQP', sparse=sparse)
        print('Sparse: ', sparse)
        print(bekk.param_final.theta)
        print('Time elapsed %.2f, seconds\n' % (time.time() - time_start))

    bekk.print_error()

    return bekk


def time_recursion():

    nstocks = 2
    nobs = 2000
    restriction = 'full'
    # A, B, C - n x n matrices
    amat = np.eye(nstocks) * .09**.5
    bmat = np.eye(nstocks) * .9**.5
    # Craw = np.ones((nstocks, nstocks))*.5 + np.eye(nstocks)*.5
    # Choose intercept to normalize unconditional variance to one
    craw = np.eye(nstocks) - amat.dot(amat) - bmat.dot(bmat)
    cmat = sl.cholesky(craw, 1)

    param_true = BEKKParams(a_mat=amat, b_mat=bmat, c_mat=cmat,
                            restriction=restriction, var_target=False)
    innov = simulate_bekk(param_true, nobs=nobs, distr='normal')


    hvar = filter_var_numba(innov, cmat, amat, bmat,
                            param_true.unconditional_var())


if __name__ == '__main__':

#    np.set_printoptions(precision=4, suppress=True)
#    nstocks = 2
#    var_target = False
#    nobs = 2000
#    restriction = 'diagonal'
#    bekk = test_bekk(nstocks=nstocks, simulate=True, var_target=var_target,
#                     restriction=restriction,
#                     nobs=nobs, log_file='../logs/log_sim.txt')
##    test_bekk(nstocks=nstocks, simulate=False, var_target=var_target,
##              nobs=nobs, log_file='log_real.txt')
#
#    print(bekk.param_true.theta)


    nstocks = 6
    nobs = 2000
    restriction = 'full'
    # A, B, C - n x n matrices
    amat = np.eye(nstocks) * .09**.5
    bmat = np.eye(nstocks) * .9**.5
    # Craw = np.ones((nstocks, nstocks))*.5 + np.eye(nstocks)*.5
    # Choose intercept to normalize unconditional variance to one
    craw = np.eye(nstocks) - amat.dot(amat) - bmat.dot(bmat)
    cmat = sl.cholesky(craw, 1)

    param_true = BEKKParams(a_mat=amat, b_mat=bmat, c_mat=cmat,
                            restriction=restriction, var_target=False)
    innov = simulate_bekk(param_true, nobs=nobs, distr='normal')

    hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
    hvar[0] = param_true.unconditional_var()

    out1 = recursion(hvar, innov, amat, bmat, cmat)
    out2 = filter_var_python(hvar, innov, amat, bmat, cmat)

    print(np.allclose(out1, out2))

    out1 = likelihood(hvar, innov)
    out2 = likelihood_python(hvar, innov)

    print(np.allclose(out1, out2))
