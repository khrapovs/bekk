#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import time

import numpy as np

from bekk import (BEKK, ParamStandard, ParamSpatial, simulate_bekk,
                  regenerate_data, plot_data)
from bekk import filter_var_python, likelihood_python
from bekk.recursion import filter_var
from bekk.likelihood import likelihood_gauss
from bekk.utils import take_time


def try_bekk():
    """Simulate and estimate BEKK model.

    """
    nstocks = 2
    use_target = True
    nobs = 2000
    restriction = 'full'
    simulate = True

    # A, B, C - n x n matrices
    A = np.eye(nstocks) * .09**.5
    B = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)

    param_true = ParamStandard.from_target(amat=A, bmat=B, target=target)
    theta_true = param_true.get_theta(use_target=use_target,
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
                      use_target=use_target, method='SLSQP', cython=cython)

        print('Cython: ', cython)
        theta_final = bekk.param_final.get_theta(restriction=restriction,
                                                 use_target=use_target)
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
    param_true = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
    cmat = param_true.cmat

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

    hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
    hvar[0] = param_true.get_uvar().copy()

    with take_time('Python recursion'):
        filter_var_python(hvar, innov, amat, bmat, cmat)
        hvar1 = hvar.copy()

    hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
    hvar[0] = param_true.get_uvar().copy()

    with take_time('Cython recursion'):
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
    """Try simulating and estimating standard BEKK.

    """
    use_target = False
    restriction = 'full'
    nstocks = 6
    nobs = 2000
    # A, B, C - n x n matrices
    amat = np.eye(nstocks) * .09**.5
    bmat = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)
    param_true = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
    print(param_true)

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

#    plot_data(innov, hvar_true)

    bekk = BEKK(innov)
    result = bekk.estimate(param_start=param_true, use_target=use_target,
                           method='SLSQP', restriction=restriction)

    print(result)

    theta_true = param_true.get_theta(use_target=use_target,
                                      restriction=restriction)
    theta_final = result.param_final.get_theta(use_target=use_target,
                                               restriction=restriction)
    norm = np.linalg.norm(theta_true - theta_final)

    print('\nParameters (true and estimated):\n',
          np.vstack([theta_true, theta_final]).T)
    print('\nEucledean norm of the difference = %.4f' % norm)


def try_spatial():
    """Try simulating and estimating spatial BEKK.

    """
    use_target = False
    cfree = True
    restriction = 'group'
    nobs = 2000
    groups = [[(0, 1), (2, 3)]]
    nstocks = np.max(groups) + 1
    ncat = len(groups)
    alpha = np.array([.1, .01])
    beta = np.array([.5, .01])
    gamma = .0
    # A, B, C - n x n matrices
    avecs = np.ones((ncat+1, nstocks)) * alpha[:, np.newaxis]**.5
    bvecs = np.ones((ncat+1, nstocks)) * beta[:, np.newaxis]**.5
    dvecs = np.ones((ncat, nstocks)) * gamma**.5
    vvec = np.ones(nstocks)

    param_true = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                        vvec=vvec, groups=groups)
    print(param_true)

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

#    plot_data(innov, hvar_true)

    bekk = BEKK(innov)
    result = bekk.estimate(param_start=param_true, use_target=use_target,
                           cfree=cfree, restriction=restriction,
                           model='spatial', groups=groups, method='SLSQP',
                           cython=True)

    print(result)

    theta_true = param_true.get_theta(use_target=use_target, cfree=cfree,
                                      restriction=restriction)
    theta_final = result.param_final.get_theta(use_target=use_target,
                                               cfree=cfree,
                                               restriction=restriction)
    norm = np.linalg.norm(theta_true - theta_final)

    print('\nParameters (true and estimated):')
    print(np.vstack([theta_true, theta_final]).T)
    print('\nEucledean norm of the difference = %.4f' % norm)


def try_spatial_combinations():
    """Try simulating spatial BEKK
    and estimating it with both spatial and standard.

    """
    use_target = False
    cfree = True
    restriction = 'full'
    nstocks = 3
    nobs = 2000
    groups = [(0, 1)]
    weights = ParamSpatial.get_weight(groups=groups, nitems=nstocks)
    ncat = weights.shape[0]
    alpha = np.array([.1, .01])
    beta = np.array([.5, .01])
    gamma = .0
    # A, B, C - n x n matrices
    avecs = np.ones((ncat+1, nstocks)) * alpha[:, np.newaxis]**.5
    bvecs = np.ones((ncat+1, nstocks)) * beta[:, np.newaxis]**.5
    dvecs = np.ones((ncat, nstocks)) * gamma**.5
    vvec = np.ones(nstocks)

    param_true = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs,
                                           dvecs=dvecs, vvec=vvec,
                                           weights=weights)
    print(param_true)

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

    bekk = BEKK(innov)

    # -------------------------------------------------------------------------
    # Estimate spatial

    result = bekk.estimate(param_start=param_true, use_target=use_target,
                           restriction=restriction, cfree=cfree,
                           model='spatial', weights=weights, method='SLSQP',
                           cython=True)

    print(result)

    theta_true = param_true.get_theta(use_target=use_target, cfree=cfree,
                                      restriction=restriction)
    theta_final = result.param_final.get_theta(use_target=use_target,
                                               cfree=cfree,
                                               restriction=restriction)
    norm = np.linalg.norm(theta_true - theta_final)

    print('\nParameters (true and estimated):\n',
          np.vstack([theta_true, theta_final]).T)
    print('\nEucledean norm of the difference = %.4f' % norm)

    # -------------------------------------------------------------------------
    # Estimate standard

    param_true = ParamStandard.from_abc(amat=param_true.amat,
                                        bmat=param_true.bmat,
                                        cmat=param_true.cmat)

    result = bekk.estimate(param_start=param_true, use_target=use_target,
                           restriction=restriction, cfree=cfree,
                           model='standard', weights=weights, method='SLSQP',
                           cython=True)

    print(result)

    theta_true = param_true.get_theta(use_target=use_target,
                                      restriction=restriction)
    theta_final = result.param_final.get_theta(use_target=use_target,
                                               restriction=restriction)
    norm = np.linalg.norm(theta_true - theta_final)

    print('\nParameters (true and estimated):\n',
          np.vstack([theta_true, theta_final]).T)
    print('\nEucledean norm of the difference = %.4f' % norm)


def try_interative_estimation_standard():
    """Try estimating parameters from simple to more complicated model.

    """
    restriction = 'scalar'
    nstocks = 3
    nobs = 2000
    # A, B, C - n x n matrices
    amat = np.eye(nstocks) * .09**.5
    bmat = np.eye(nstocks) * .9**.5
    target = np.eye(nstocks)
    param_true = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

    bekk = BEKK(innov)
    result = bekk.estimate(use_target=False, restriction=restriction)

    print(result)


def try_interative_estimation_spatial():
    """Try estimating parameters from simple to more complicated model.

    """
    restriction = 'full'
    nstocks = 3
    nobs = 2000
    groups = [(0, 1)]
    weights = get_weight(groups=groups, nitems=nstocks)
    ncat = weights.shape[0]
    alpha = np.array([.1, .01])
    beta = np.array([.5, .01])
    gamma = .0
    # A, B, C - n x n matrices
    avecs = np.ones((ncat+1, nstocks)) * alpha[:, np.newaxis]**.5
    bvecs = np.ones((ncat+1, nstocks)) * beta[:, np.newaxis]**.5
    dvecs = np.ones((ncat, nstocks)) * gamma**.5
    vvec = np.ones(nstocks)

    param_true = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs,
                                           dvecs=dvecs, vvec=vvec,
                                           weights=weights)

    innov, hvar_true = simulate_bekk(param_true, nobs=nobs, distr='normal')

    bekk = BEKK(innov)
    result = bekk.estimate(use_target=False, restriction=restriction,
                           cfree=False, model='spatial', weights=weights)
    print(result)


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)

#    try_bekk()

#    time_likelihood()

#    with take_time('\nTotal simulation and estimation'):
#        try_standard()

    with take_time('Total simulation and estimation'):
        try_spatial()

#    with take_time('Total simulation and estimation'):
#        try_spatial_combinations()

#    with take_time('Initialize parameters for standard model'):
#        try_interative_estimation_standard()

#    with take_time('Initialize parameters for spatial model'):
#        try_interative_estimation_spatial()
