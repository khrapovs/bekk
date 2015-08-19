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
    hvar
        Condiational variance

    Methods
    -------
    estimate
        Estimate parameters of the model
    evaluate_forecast
        Evaluate forecast using rolling window

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

    def likelihood(self, theta, model='standard', restriction='full',
                   target=None, cfree=False, groups=None, cython=True):
        """Compute the conditional log-likelihood function.

        Parameters
        ----------
        theta : 1dim array
            Dimension depends on the model restriction
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

        target : (nstocks, nstocks) array
            Estimate of unconditional variance matrix
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        groups : list of lists of tuples
            Encoded groups of items
        cython : bool
            Whether to use Cython optimizations (True) or not (False)

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
                                            groups=groups)
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
                 use_target=True, model='standard', groups=None,
                 method='SLSQP', cython=True):
        """Estimate parameters of the BEKK model.

        Parameters
        ----------
        param_start : ParamStandard or ParamSpatial instance
            Starting parameters. See Notes for more details.
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
                - 'group' (only applicable with 'spatial' model)
                - 'scalar'

        use_target : bool
            Whether to use variance targeting (True) or not (False)
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        groups : list of lists of tuples
            Encoded groups of items
        method : str
            Optimization method. See scipy.optimize.minimize
        cython : bool
            Whether to use Cython optimizations (True) or not (False)

        Returns
        -------
        BEKKResults instance
            Estimation results object

        Notes
        -----

        If no param_start is given, the program will estimate parameters in
        the order 'from simple to more complicated' (from scalar to diagonal
        to full) while always using variance targeting.

        """
        # Check for incompatible inputs
        if use_target and cfree:
            raise ValueError('use_target and cfree are incompatible!')
        if (groups is not None) and (model != 'spatial'):
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
                                                      groups=groups)
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
                  'restriction': restriction, 'groups': groups,
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
                                                  groups=groups)
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

        kwargs = {'use_target': True, 'model': 'standard'}
        est_partial = partial(self.estimate, **kwargs)

        if restriction in ('diagonal', 'full', 'group', 'scalar'):
            with take_time('Estimating scalar'):
                result = est_partial(param_start=param, restriction='scalar')
                param = result.param_final

        if restriction in ('diagonal', 'full'):
            with take_time('Estimating diagonal'):
                result = est_partial(param_start=param, restriction='diagonal')
                param = result.param_final

        if restriction == 'full':
            with take_time('Estimating full'):
                result = est_partial(param_start=param, restriction='full')
                param = result.param_final

        return param

    def init_param_spatial(self, restriction='scalar', groups=None):
        """Estimate scalar BEKK with variance targeting.

        Parameters
        ----------
        restriction : str
            Restriction on parameters.

            Must be
                - 'full' =  'diagonal'
                - 'group'
                - 'scalar'

        groups : list of lists of tuples
            Encoded groups of items

        Returns
        -------
        ParamSpatial instance
            Parameter object

        """
        param = ParamSpatial(nstocks=self.innov.shape[1])

        kwargs = {'use_target': True, 'groups': groups, 'model': 'spatial'}
        est_partial = partial(self.estimate, **kwargs)

        if restriction in ('diagonal', 'full', 'group', 'scalar'):
            with take_time('Estimating scalar'):
                result = est_partial(param_start=param, restriction='scalar')
                param = result.param_final

        if restriction in ('diagonal', 'full', 'group'):
            with take_time('Estimating group'):
                result = est_partial(param_start=param, restriction='group')
                param = result.param_final

        if restriction in ('diagonal', 'full'):
            with take_time('Estimating full/diagonal'):
                result = est_partial(param_start=param, restriction='full')
                param = result.param_final

        return param

    @staticmethod
    def forecast(hvar=None, innov=None, param=None):
        """One step ahead volatility forecast.

        Parameters
        ----------
        hvar : (nstocks, nstocks) array
            Current variance/covariances
        innov : (nstocks, ) array
            Current inovations
        param : ParamStandard or ParamSpatial instance
            Parameter object

        Returns
        -------
        (nstocks, nstocks) array
            Volatility forecast

        """
        forecast = param.cmat.dot(param.cmat.T)
        forecast += param.amat.dot(BEKK.sqinnov(innov)).dot(param.amat.T)
        forecast += param.bmat.dot(hvar).dot(param.bmat.T)
        return forecast

    @staticmethod
    def sqinnov(innov):
        """Volatility proxy. Square returns.

        Parameters
        ----------
        innov : (nstocks, ) array
            Current inovations

        Returns
        -------
        (nstocks, nstocks) array
            Volatility proxy

        """
        return innov * innov[:, np.newaxis]

    @staticmethod
    def loss_frob(forecast=None, proxy=None):
        """One step ahead volatility forecast.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility

        Returns
        -------
        float
            loss_frob function

        """
        return np.linalg.norm(forecast - proxy) / forecast.shape[0]**2

    @staticmethod
    def evaluate_forecast(param_start=None, innov_all=None, window=1000,
                          model='standard', use_target=True, groups=None,
                          restriction='scalar'):
        """Evaluate forecast using rolling window.

        Parameters
        ----------
        param_start : ParamStandard or ParamSpatial instance
            Initial parameters for estimation
        innov_all: (nobs, nstocks) array
            Inovations
        window : int
            Window length for in-sample estimation
        model : str
            Specific model to estimate.

            Must be
                - 'standard'
                - 'spatial'

        restriction : str
            Restriction on parameters.

            Must be
                - 'full' =  'diagonal'
                - 'group' (for 'spatial' model only)
                - 'scalar'

        groups : list of lists of tuples
            Encoded groups of items
        use_target : bool
            Whether to use variance targeting (True) or not (False)

        Returns
        -------
        float
            Average loss_frob function

        """
        nobs = innov_all.shape[0]
        loss_frob = np.zeros(nobs - window)
        for first in range(nobs - window):
            last = window + first
            innov = innov_all[first:last]
            bekk = BEKK(innov)
            result = bekk.estimate(param_start=param_start, groups=groups,
                                   use_target=use_target, model=model,
                                   restriction=restriction)
            forecast = BEKK.forecast(hvar=result.hvar[-1], innov=innov[-1],
                                     param=result.param_final)
            proxy = BEKK.sqinnov(innov_all[last])
            loss_frob[first] = BEKK.loss_frob(forecast=forecast, proxy=proxy)
        return np.mean(loss_frob)
