#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
BEKK estimation
===============

Estimation is performed using Quasi Maximum Likelihood (QML) method.
Specifically, the individual contribution to the Gaussian log-likelihood is

.. math::
    l_{t}\left(\theta\right)=
    -\ln\left|H_{t}\right|-u_{t}^{\prime}H_{t}^{-1}u_{t}.

"""
from __future__ import print_function, division

import time

import numpy as np

from scipy.optimize import minimize
from functools import partial

from bekk import ParamStandard, ParamSpatial, BEKKResults
from .utils import (estimate_uvar, likelihood_python, filter_var_python,
                    take_time)
try:
    from .recursion import filter_var
    from .likelihood import likelihood_gauss
except:
    print('Failed to import cython modules. '
          + 'Temporary hack to compile documentation.')

__all__ = ['BEKK']


class BEKK(object):

    """BEKK estimation class.

    Attributes
    ----------
    innov
        Return innovations
    log_file
        File name to write the results of estimation
    param_start
        Initial values of model parameters
    param_final
        Final values of model parameters
    opt_out
        Optimization results

    Methods
    -------
    estimate
        Estimate parameters of the model

    """

    def __init__(self, innov):
        """Initialize the class.

        Parameters
        ----------
        innov : (nobs, nstocks) array
            Return innovations

        """
        self.innov = innov
        self.hvar = None

    def likelihood(self, theta, model='standard', target=None, cfree=False,
                   restriction='full', weights=None, cython=True):
        """Compute the conditional log-likelihood function.

        Parameters
        ----------
        theta : 1dim array
            Dimension depends on the model restriction

        Returns
        -------
        float
            The value of the minus log-likelihood function.
            If some regularity conditions are violated, then it returns
            some obscene number.

        """
        if model == 'standard':
            param = ParamStandard.from_theta(theta=theta, target=target,
                                             cfree=cfree,
                                             nstocks=self.innov.shape[1],
                                             restriction=restriction)
        elif model == 'spatial':
            param = ParamSpatial.from_theta(theta=theta, target=target,
                                            cfree=cfree,
                                            restriction=restriction,
                                            weights=weights)
        else:
            raise NotImplementedError('The model is not implemented!')

        if param.constraint() >= 1 or param.cmat is None:
            return 1e10

        args = [self.hvar, self.innov, param.amat, param.bmat, param.cmat]

        if cython:
            filter_var(*args)
            return likelihood_gauss(self.hvar, self.innov)
        else:
            filter_var_python(*args)
            return likelihood_python(self.hvar, self.innov)

    def estimate(self, param_start=None, restriction='scalar', cfree=False,
                 use_target=True, model='standard', weights=None,
                 method='SLSQP', cython=True):
        """Estimate parameters of the BEKK model.

        Parameters
        ----------
        param_start : BEKKParams instance
            Starting parameters
        model : str
            Specific model to estimate.

            Must be
                - 'standard'
                - 'spatial'

        restriction : str
            Restriction on parameters.

            Must be
                - 'full'
                - 'diagonal'
                - 'scalar'

        use_target : bool
            Whether to use variance targeting (True) or not (False)
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        weights : (ncat, nstocks, nstocks) array
            Weight matrices for spatial only
        method : str
            Optimization method. See scipy.optimize.minimize
        cython : bool
            Whether to use Cython optimizations (True) or not (False)

        Returns
        -------
        BEKKResults instance
            Estimation results object

        """
        # Check for incompatible inputs
        if use_target and cfree:
            raise ValueError('use_target and cfree are incompatible!')
        if (weights is not None) and (model != 'spatial'):
            raise ValueError('The model is incompatible with weights!')
        # Update default settings
        nobs, nstocks = self.innov.shape
        var_target = estimate_uvar(self.innov)
        self.hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        self.hvar[0] = var_target.copy()

        # Check for existence of initial guess among arguments.
        # Otherwise, initialize.
        if param_start is None:
            if model == 'standard':
                param_start = self.init_param_standard(restriction=restriction)
            elif model == 'spatial':
                param_start = self.init_param_spatial(restriction=restriction,
                                                      weights=weights)
            else:
                raise NotImplementedError('The model is not implemented!')

        # Get vector of parameters to start optimization
        theta_start = param_start.get_theta(restriction=restriction,
                                            use_target=use_target, cfree=cfree)
        if use_target:
            target = var_target
        else:
            target = None

        # Optimization options
        options = {'disp': False, 'maxiter': int(1e6)}
        # Likelihood arguments
        kwargs = {'model': model, 'target': target, 'cfree': cfree,
                  'restriction': restriction, 'weights': weights,
                  'cython': cython}
        # Likelihood function
        likelihood = partial(self.likelihood, **kwargs)
        # Start timer for the whole optimization
        time_start = time.time()
        # Run optimization
        opt_out = minimize(likelihood, theta_start,
                           method=method, options=options)
        # How much time did it take in minutes?
        time_delta = time.time() - time_start

        # Store optimal parameters in the corresponding class
        if model == 'standard':
            param_final = ParamStandard.from_theta(theta=opt_out.x,
                                                   restriction=restriction,
                                                   target=target, cfree=cfree,
                                                   nstocks=nstocks)
        elif model == 'spatial':
            param_final = ParamSpatial.from_theta(theta=opt_out.x,
                                                  restriction=restriction,
                                                  target=target, cfree=cfree,
                                                  weights=weights)
        else:
            raise NotImplementedError('The model is not implemented!')

        return BEKKResults(innov=self.innov, hvar=self.hvar, cython=cython,
                           var_target=var_target, model=model, method=method,
                           use_target=use_target, cfree=cfree,
                           restriction=restriction,
                           param_start=param_start, param_final=param_final,
                           time_delta=time_delta, opt_out=opt_out)

    def init_param_standard(self, restriction='scalar'):
        """Estimate scalar BEKK with variance targeting.

        Parameters
        ----------
        restriction : str
            Restriction on parameters.

            Must be
                - 'full'
                - 'diagonal'
                - 'scalar'

        Returns
        -------
        ParamStandard instance
            Parameter object

        """
        param = ParamStandard(nstocks=self.innov.shape[1])

        with take_time('Estimating scalar'):
            result = self.estimate(param_start=param, use_target=True,
                                   restriction='scalar', model='standard')
        param = result.param_final

        if restriction in ('diagonal', 'full'):
            with take_time('Estimating diagonal'):
                result = self.estimate(param_start=param, use_target=True,
                                       restriction='diagonal',
                                       model='standard')
            param = result.param_final

        if restriction == 'full':
            with take_time('Estimating full'):
                result = self.estimate(param_start=param, use_target=True,
                                       restriction='full', model='standard')
            param = result.param_final

        return param

    def init_param_spatial(self, restriction='scalar', weights=None):
        """Estimate scalar BEKK with variance targeting.

        Parameters
        ----------
        restriction : str
            Restriction on parameters.

            Must be
                - 'full'
                - 'diagonal'
                - 'scalar'
        weights : (ncat, nstocks, nstocks) array
            Weight matrices for spatial only

        Returns
        -------
        ParamSpatial instance
            Parameter object

        """
        param = ParamSpatial(nstocks=self.innov.shape[1])

        with take_time('Estimating scalar'):
            result = self.estimate(param_start=param, use_target=True,
                                   weights=weights,
                                   restriction='scalar', model='spatial')
        param = result.param_final

        if restriction in ('diagonal', 'full'):
            with take_time('Estimating full/diagonal'):
                result = self.estimate(param_start=param, use_target=True,
                                       weights=weights,
                                       restriction='full', model='spatial')
            param = result.param_final

        return param
