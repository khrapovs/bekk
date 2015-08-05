#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import time

import numpy as np
import scipy.linalg as scl

from bekk import BEKK, BEKKParams, simulate_bekk, regenerate_data, plot_data
from bekk import filter_var_python, likelihood_python
from bekk.recursion import filter_var
from bekk.likelihood import likelihood_gauss
from bekk.utils import take_time, estimate_h0


def try_bekk(nstocks=2, nobs=500, restriction='scalar', var_target=True,
              simulate=True, log_file=None):
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
    if log_file is not None:
        with open(log_file, 'w') as texfile:
            texfile.write('')

    # A, B, C - n x n matrices
    A = np.eye(nstocks) * .09**.5
    B = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)

    param_true = BEKKParams.from_target(amat=A, bmat=B, target=target)
    theta_true = param_true.get_theta(var_target=var_target,
                                      restriction=restriction)
    print('True parameter:\n', theta_true)
    # Data file
    innov_file = '../data/innovations.npy'

    if simulate:
        # Simulate data
        innov, hvar_true = simulate_bekk(param_true, nobs=nobs,
                                         distr='skewt', degf=30)
        np.savetxt(innov_file[:-4] + '.csv', innov, delimiter=",")

    else:
        # Regenerate real data
        regenerate_data(innov_file=innov_file, nstocks=nstocks, nobs=nobs)
        # Load data from the drive
        innov = np.load(innov_file)

    # Estimate parameters
    params = []
    for cython in [True, False]:
        time_start = time.time()
        # Initialize the object
        bekk = BEKK(innov)
        bekk.estimate(param_start=param_true, restriction=restriction,
                      var_target=var_target, method='SLSQP', cython=cython)

        print('Cython: ', cython)
        theta_final = bekk.param_final.get_theta(restriction=restriction,
                                                 var_target=var_target)
        print(theta_final)
        params.append(theta_final)
        print('Time elapsed %.2f, seconds\n' % (time.time() - time_start))

    print('\nNorm difference between the estimates: %.4f'
        % np.linalg.norm(params[0] - params[1]))
    return bekk


def time_likelihood():
    """Compare speeds of recrsions and likelihoods.

    """
    nstocks = 2
    nobs = 2000
    # A, B, C - n x n matrices
    amat = np.eye(nstocks) * .09**.5
    bmat = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)
    param_true = BEKKParams.from_target(amat=amat, bmat=bmat, target=target)
    cmat = param_true.cmat

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

    hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
    hvar[0] = param_true.get_uvar().copy()

    with take_time('Python recursion'):
        filter_var_python(hvar, innov, amat, bmat, cmat)
        hvar1 = hvar.copy()

    hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
    hvar[0] = param_true.get_uvar().copy()

    with take_time('Cython recursion 2'):
        filter_var(hvar, innov, amat, bmat, cmat)
        hvar2 = hvar.copy()
        idxl = np.tril_indices(nstocks)
        idxu = np.triu_indices(nstocks)
        hvar2[:, idxu[0], idxu[1]] = hvar2[:, idxl[0], idxl[1]]

    print(np.allclose(hvar_true, hvar1))
    print(np.allclose(hvar_true, hvar2))

    with take_time('Python likelihood'):
        out1 = likelihood_python(hvar, innov)
    with take_time('Cython likelihood'):
        out2 = likelihood_gauss(hvar, innov)

    print(np.allclose(out1, out2))


def try_standard():
    """Try simulating and estimating spatial BEKK.

    """
    var_target = False
    nstocks = 2
    nobs = 2000
    # A, B, C - n x n matrices
    amat = np.eye(nstocks) * .09**.5
    bmat = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)
    param_true = BEKKParams.from_target(amat=amat, bmat=bmat, target=target)

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

#    plot_data(innov, hvar_true)

    bekk = BEKK(innov)
    bekk.estimate(param_start=param_true, var_target=var_target,
                  method='SLSQP', cython=True)

    print('Target:\n', estimate_h0(innov))

    print('\nTrue parameters:\n', param_true)
    print('\nEstimated parameters:\n', bekk.param_final)

    print('\nTrue parameters:\n',
          param_true.get_theta(var_target=var_target))
    print('\nEstimated parameters:\n',
          bekk.param_final.get_theta(var_target=var_target))


def try_spatial():
    """Try simulating and estimating spatial BEKK.

    """
    var_target = False
    nstocks = 3
    nobs = 2000
    weights = np.array([[[0, .1, 0], [.1, 0, 0], [0, 0, 0]]])
    ncat = weights.shape[0]
    alpha = np.array([.1, .05])
    beta = np.array([.8, .1])
    gamma = .005
    # A, B, C - n x n matrices
    avecs = np.ones((ncat+1, nstocks)) * alpha[:, np.newaxis]**.5
    bvecs = np.ones((ncat+1, nstocks)) * beta[:, np.newaxis]**.5
    dvecs = np.ones((ncat, nstocks)) * gamma**.5
    vvec = np.ones(nstocks)

    param = BEKKParams.from_spatial(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                    vvec=vvec, weights=weights)

    innov, hvar_true = simulate_bekk(param, nobs=nobs, distr='normal')

#    plot_data(innov, hvar_true)

    bekk = BEKK(innov)
    bekk.estimate_spatial(param_start=param, var_target=var_target,
                          weights=weights, method='SLSQP', cython=True)

    print('Target:\n', estimate_h0(innov))

    print('\nTrue parameters:\n', param)
    print('\nEstimated parameters:\n', bekk.param_final)

    print('\nTrue parameters:\n',
          param.get_theta_spatial(var_target=var_target))
    print('\nEstimated parameters:\n',
          bekk.param_final.get_theta_spatial(var_target=var_target))

    print('\nTrue a:\n', param.avecs)
    print('\nEstimated a:\n', bekk.param_final.avecs)

    print('\nTrue b:\n', param.bvecs)
    print('\nEstimated b:\n', bekk.param_final.bvecs)

    print('\nTrue d:\n', param.dvecs)
    print('\nEstimated d:\n', bekk.param_final.dvecs)

    print('\nTrue v:\n', param.vvec)
    print('\nEstimated v:\n', bekk.param_final.vvec)


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    nstocks = 2
    var_target = True
    nobs = 2000
    restriction = 'full'

#    bekk = try_bekk(nstocks=nstocks, simulate=True, var_target=var_target,
#                     restriction=restriction, nobs=nobs)

#    try_bekk(nstocks=nstocks, simulate=False, var_target=var_target,
#              nobs=nobs, log_file='log_real.txt')

#    time_likelihood()

#    with take_time('Estimation'):
#        try_standard()

    with take_time('Estimation'):
        try_spatial()
