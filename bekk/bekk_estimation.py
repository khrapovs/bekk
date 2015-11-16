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
import itertools

import pandas as pd
import numpy as np
import scipy.linalg as scl

from scipy.optimize import minimize, basinhopping
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
                   target=None, cfree=False, groups=None, cython=True,
                   use_penalty=False):
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
        use_penalty : bool
            Whether to include penalty term in the likelihood

        Returns
        -------
        float
            The value of the minus log-likelihood function.
            If some regularity conditions are violated, then it returns
            some obscene number.

        """
        try:
            if model == 'standard':
                param = ParamStandard.from_theta(theta=theta, target=target,
                                                 nstocks=self.innov.shape[1],
                                                 restriction=restriction)
            elif model == 'spatial':
                param = ParamSpatial.from_theta(theta=theta, target=target,
                                                cfree=cfree,
                                                restriction=restriction,
                                                groups=groups)
            else:
                raise NotImplementedError('The model is not implemented!')

            # TODO: Temporary hack to exclude errors in optimization
            if isinstance(param, np.ndarray):
                return 1e10
            if param.constraint() >= 1:
                return 1e10
            # if param.uvar_bad():
            #     return 1e10

            args = [self.hvar, self.innov, param.amat, param.bmat, param.cmat]

            penalty = param.penalty() if use_penalty else 0

            if cython:
                filter_var(*args)
                return likelihood_gauss(self.hvar, self.innov) + penalty
            else:
                filter_var_python(*args)
                return likelihood_python(self.hvar, self.innov) + penalty
        except:
            return 1e10

    def estimate(self, param_start=None, restriction='scalar', cfree=False,
                 use_target=False, model='standard', groups=None,
                 method='SLSQP', cython=True, use_penalty=False):
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
        use_penalty : bool
            Whether to include penalty term in the likelihood

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
        # Start timer for the whole optimization
        time_start = time.time()

        # Check for incompatible inputs
        if use_target and cfree:
            raise ValueError('use_target and cfree are incompatible!')
#        if (groups is not None) and (model != 'spatial'):
#            raise ValueError('The model is incompatible with weights!')
        # Update default settings
        nobs, nstocks = self.innov.shape
        var_target = estimate_uvar(self.innov)
        self.hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        self.hvar[0] = var_target.copy()

        # Check for existence of initial guess among arguments.
        # Otherwise, initialize.
        if param_start is None:
            common = {'restriction': restriction, 'method': method,
                      'use_penalty': use_penalty}
            if model == 'standard':
                param_start = self.init_param_standard(**common)
            elif model == 'spatial':
                param_start = self.init_param_spatial(groups=groups,
                                                      cfree=cfree, **common)
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
        if method == 'Nelder-Mead':
            options['maxfev'] = 3000
        # Likelihood arguments
        kwargs = {'model': model, 'target': target, 'cfree': cfree,
                  'restriction': restriction, 'groups': groups,
                  'cython': cython, 'use_penalty': use_penalty}
        # Likelihood function
        likelihood = partial(self.likelihood, **kwargs)

        # Run optimization
        if method == 'basin':
            opt_out = basinhopping(likelihood, theta_start, niter=100,
                                   disp=options['disp'],
                                   minimizer_kwargs={'method': 'Nelder-Mead'})
        else:
            opt_out = minimize(likelihood, theta_start,
                               method=method, options=options)
        # How much time did it take in minutes?
        time_delta = time.time() - time_start

        # Store optimal parameters in the corresponding class
        if model == 'standard':
            param_final = ParamStandard.from_theta(theta=opt_out.x,
                                                   restriction=restriction,
                                                   target=target,
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

    def init_param_standard(self, restriction='scalar',
                            method='SLSQP', use_penalty=False):
        """Estimate scalar BEKK with variance targeting.

        Parameters
        ----------
        restriction : str
            Restriction on parameters.

            Must be
                - 'full'
                - 'diagonal'
                - 'scalar'

        method : str
            Optimization method. See scipy.optimize.minimize

        Returns
        -------
        ParamStandard instance
            Parameter object

        """
        param = ParamStandard(nstocks=self.innov.shape[1],
                              target=estimate_uvar(self.innov),
                              abstart=(.2, .6))

        if restriction == 'scalar':
            return param

        kwargs = {'model': 'standard', 'use_penalty': use_penalty,
                  'use_target': False, 'method': method}
        est_partial = partial(self.estimate, **kwargs)

        if restriction in ('diagonal', 'full', 'group', 'scalar'):
            result = est_partial(param_start=param, restriction='scalar')
            param = result.param_final

        if restriction in ('diagonal', 'full'):
            result = est_partial(param_start=param, restriction='diagonal')
            param = result.param_final

        return param

    def init_param_spatial(self, restriction='scalar', groups=None,
                           method='SLSQP', cfree=False, use_penalty=False):
        """Estimate scalar BEKK with variance targeting.

        Parameters
        ----------
        restriction : str
            Restriction on parameters.

            Must be
                - 'full' =  'diagonal'
                - 'group'
                - 'scalar'

        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        groups : list of lists of tuples
            Encoded groups of items
        method : str
            Optimization method. See scipy.optimize.minimize

        Returns
        -------
        ParamSpatial instance
            Parameter object

        """
        param = ParamSpatial.from_groups(groups=groups,
                                         target=estimate_uvar(self.innov),
                                         abstart=(.2, .7))

        if restriction == 'scalar':
            return param

        kwargs = {'use_target': False, 'groups': groups,
                  'use_penalty': use_penalty, 'model': 'spatial',
                  'cfree': cfree, 'method': method}
        est_partial = partial(self.estimate, **kwargs)

        if restriction in ('diagonal', 'full', 'group', 'scalar'):
            result = est_partial(param_start=param, restriction='scalar')
            param = result.param_final

        if restriction in ('diagonal', 'full', 'group'):
            result = est_partial(param_start=param, restriction='group')
            param = result.param_final

        return param

    def estimate_loop(self, model='standard', use_target=True, groups=None,
                      restriction='scalar', cfree=False,
                      method='SLSQP', ngrid=2, use_penalty=False):
        """Estimate parameters starting from a grid of a and b.

        Parameters
        ----------
        model : str
            Specific model to estimate.

            Must be
                - 'standard'
                - 'spatial'

        restriction : str
            Restriction on parameters.

            Must be
                - 'full' =  'diagonal'
                - 'group'
                - 'scalar'

        groups : list of lists of tuples
            Encoded groups of items
        use_target : bool
            Whether to use variance targeting (True) or not (False)
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        method : str
            Optimization method. See scipy.optimize.minimize
        ngrid : int
            Number of starting values in one dimension
        use_penalty : bool
            Whether to include penalty term in the likelihood

        Returns
        -------
        BEKKResults instance
            Estimation results object

        """
        target = estimate_uvar(self.innov)
        nstocks = self.innov.shape[1]
        achoice = np.linspace(.01, .5, ngrid)
        bchoice = np.linspace(.1, .9, ngrid)
        out = dict()
        for abstart in itertools.product(achoice, bchoice):
            if model == 'spatial':
                param = ParamSpatial.from_groups(groups=groups,
                                                 target=target,
                                                 abstart=abstart)
            if model == 'standard':
                param = ParamStandard(nstocks=nstocks, target=target,
                                      abstart=abstart)
            if param.constraint() >= 1:
                continue
            result = self.estimate(param_start=param, method=method,
                                   use_target=use_target, cfree=cfree,
                                   model=model, restriction=restriction,
                                   groups=groups, use_penalty=use_penalty)
            out[abstart] = (result.opt_out.fun, result)

        df = pd.DataFrame.from_dict(out, orient='index')
        return df.sort_values(by=0).iloc[0, 1]

    @staticmethod
    def forecast_one(hvar=None, innov=None, param=None):
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
    def loss_eucl(forecast=None, proxy=None):
        """Eucledean loss function.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility

        Returns
        -------
        float

        """
        diff = (forecast - proxy)[np.tril_indices_from(forecast)]
        return np.linalg.norm(diff)**2

    @staticmethod
    def loss_frob(forecast=None, proxy=None):
        """Frobenius loss function.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility

        Returns
        -------
        float

        """
        diff = forecast - proxy
        return np.trace(diff.T.dot(diff))

    @staticmethod
    def loss_stein(forecast=None, proxy=None):
        """Stein loss function for non-degenerate proxy.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility

        Returns
        -------
        float

        """
        nstocks = forecast.shape[0]
        ratio = np.linalg.solve(forecast, proxy)
        return np.trace(ratio) - np.log(np.linalg.det(ratio)) - nstocks

    @staticmethod
    def loss_stein2(forecast=None, innov=None):
        """Stein loss function.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        innov : (nstocks, ) array
            Returns

        Returns
        -------
        float

        """
        lower = True
        forecast, lower = scl.cho_factor(forecast, lower=lower,
                                         check_finite=False)
        norm_innov = scl.cho_solve((forecast, lower), innov,
                                   check_finite=False)
        return (np.log(np.diag(forecast)**2) + norm_innov * innov).sum()

    @staticmethod
    def collect_losses(param_start=None, innov_all=None, window=1000,
                       model='standard', use_target=False, groups=None,
                       restriction='scalar', cfree=False, method='SLSQP',
                       use_penalty=False, ngrid=5, tname=None):
        """Collect forecast losses using rolling window.

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
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        ngrid : int
            Number of starting values in one dimension
        use_penalty : bool
            Whether to include penalty term in the likelihood

        Returns
        -------
        float
            Average loss_frob function

        """
        nobs = innov_all.shape[0]
        logl = np.zeros(nobs - window)
        loss_eucl = np.zeros(nobs - window)
        loss_frob = np.zeros(nobs - window)
        loss_stein = np.zeros(nobs - window)
        time_delta = np.zeros(nobs - window)
        loop = np.zeros(nobs - window)

        common = {'groups': groups[1], 'use_target': use_target,
                  'model': model, 'restriction': restriction, 'cfree': cfree,
                  'use_penalty': use_penalty}

        loc_name = tname + '_' + model +'_' + restriction + '_' + groups[0]
        fname = '../data/losses/' + loc_name + '.h5'

        for first in range(nobs - window):
            last = window + first
            innov = innov_all[first:last]
            bekk = BEKK(innov)

            time_start = time.time()
            if first == 0:
                result = bekk.estimate(method=method, **common)
            else:
                result = bekk.estimate(param_start=param_start,
                                       method=method, **common)

#            if first == 0:
#                result = bekk.estimate_loop(ngrid=ngrid, **common)
#                result = bekk.estimate(param_start=result.param_final,
#                                       method='basin', **common)
#            else:
#                result = bekk.estimate(param_start=param_start, method=method,
#                                       **common)
#
            if result.opt_out.fun == 1e10:
                loop[first] = 1
                result = bekk.estimate(param_start=param_start,
                                       method='basin', **common)
            if result.opt_out.fun == 1e10:
                loop[first] = 2
                result = bekk.estimate_loop(ngrid=ngrid, method=method,
                                            **common)

            time_delta[first] = time.time() - time_start

            param_start = result.param_final

            forecast = BEKK.forecast_one(hvar=result.hvar[-1], innov=innov[-1],
                                         param=result.param_final)
            proxy = BEKK.sqinnov(innov_all[last])

            logl[first] = result.opt_out.fun
            loss_eucl[first] = BEKK.loss_eucl(forecast=forecast, proxy=proxy)
            loss_frob[first] = BEKK.loss_frob(forecast=forecast, proxy=proxy)
            loss_stein[first] = BEKK.loss_stein2(forecast=forecast,
                                                 innov=innov_all[last])

            data = {'eucl': loss_eucl[first], 'frob': loss_frob[first],
                    'stein': loss_stein[first], 'logl': logl[first],
                    'time_delta': time_delta[first], 'loop': loop[first]}

            ids = [model, restriction, groups[0], first]
            names = ['model', 'restriction', 'group', 'first']
            index = pd.MultiIndex.from_arrays(ids, names=names)
            losses = pd.DataFrame(data, index=index)

            append = False if first == 0 else True
            losses.to_hdf(fname, tname, format='t', append=append,
                          min_itemsize=10)

        return {'logl': logl, 'eucl': loss_eucl,
                'frob': loss_frob, 'stein': loss_stein,
                'time_delta': time_delta, 'loop': loop}
