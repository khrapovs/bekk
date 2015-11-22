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
import scipy.stats as scs

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
    def weights_equal(nstocks):
        """Equal weights.

        Parameters
        ----------
        nstocks : int
            Number of stocks

        Returns
        -------
        (nstocks, ) array

        """
        return np.ones(nstocks) / nstocks

    @staticmethod
    def weights_minvar(hvar):
        """Minimum variance weights.

        Returns
        -------
        (nobs, nstocks) array

        """
        nstocks = hvar.shape[0]
        inv_hvar = np.linalg.solve(hvar, np.ones(nstocks))
        return inv_hvar / inv_hvar.sum()

    @staticmethod
    def weights(nstocks=None, hvar=None, kind='equal'):
        """Portfolio weights.

        Parameters
        ----------
        nstocks : int
            Number of stocks
        weight : str
            Either 'equal' or 'minvar' (minimum variance).

        Returns
        -------
        (nobs, nstocks) array

        """
        if kind == 'equal':
            return BEKK.weights_equal(nstocks)
        elif kind == 'minvar':
            return BEKK.weights_minvar(hvar)
        else:
            raise ValueError('Weight choice is not supported!')

    @staticmethod
    def pret(innov, weights=None):
        """Portfolio return.

        Parameters
        ----------
        innov : (nstocks, ) array
            Current inovations
        weights : (nstocks, ) array
            Portfolio weightings

        Returns
        -------
        float
            Portfolio return

        """
        if weights is None:
            nstocks = innov.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        return np.sum(innov * weights)

    @staticmethod
    def pvar(var, weights=None):
        """Portfolio variance.

        Parameters
        ----------
        var : (nstocks, nstocks) array
            Variance matrix of returns
        weights : (nstocks, ) array
            Portfolio weightings

        Returns
        -------
        float
            Portfolio variance

        """
        if weights is None:
            nstocks = var.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        return np.sum(weights * var.dot(weights))

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
    def portf_lscore(forecast=None, innov=None, weights=None):
        """Portfolio log-score loss function.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        innov : (nstocks, ) array
            Returns
        weights : (nstocks, ) array
            Portfolio weights

        Returns
        -------
        float

        """
        if weights is None:
            nstocks = forecast.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        pret = BEKK.pret(innov, weights=weights)
        pvar = BEKK.pvar(forecast, weights=weights)
        return (np.log(pvar) + pret**2 / pvar) / 2

    @staticmethod
    def portf_mse(forecast=None, proxy=None, weights=None):
        """Portfolio MSE loss function.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility
        weights : (nstocks, ) array
            Portfolio weights

        Returns
        -------
        float

        """
        if weights is None:
            nstocks = forecast.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        pvar_exp = BEKK.pvar(forecast, weights=weights)
        pvar_real = BEKK.pvar(proxy, weights=weights)
        return (pvar_exp - pvar_real) ** 2

    @staticmethod
    def portf_qlike(forecast=None, proxy=None, weights=None):
        """Portfolio QLIKE loss function.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility
        weights : (nstocks, ) array
            Portfolio weights

        Returns
        -------
        float

        """
        if weights is None:
            nstocks = forecast.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        pvar_exp = BEKK.pvar(forecast, weights=weights)
        pvar_real = BEKK.pvar(proxy, weights=weights)
        return np.log(pvar_exp) + pvar_real**2 / pvar_exp

    @staticmethod
    def portf_var(forecast=None, alpha=.05, weights=None):
        """Portfolio Value-at-Risk.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        alpha : float
            Risk level. Usually 1% or 5%.
        weights : (nstocks, ) array
            Portfolio weights

        Returns
        -------
        float

        """
        if weights is None:
            nstocks = forecast.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        return scs.norm.ppf(alpha) * BEKK.pvar(forecast, weights=weights)**.5

    @staticmethod
    def var_exception(innov=None, forecast=None, alpha=.05, weights=None):
        """Exception associated with portfolio Value-at-Risk.

        Parameters
        ----------
        innov : (nstocks, ) array
            Returns
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        alpha : float
            Risk level. Usually 1% or 5%.
        weights : (nstocks, ) array
            Portfolio weights

        Returns
        -------
        float

        """
        if weights is None:
            nstocks = forecast.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        var = BEKK.portf_var(forecast=forecast, alpha=alpha, weights=weights)
        pret = BEKK.pret(innov, weights=weights)
        diff = pret - var
        if diff < 0:
            return 1
        else:
            return 0

    @staticmethod
    def loss_var(innov=None, forecast=None, alpha=.05, weights=None):
        """Loss associated with portfolio Value-at-Risk.

        Parameters
        ----------
        innov : (nstocks, ) array
            Returns
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        alpha : float
            Risk level. Usually 1% or 5%.
        weights : (nstocks, ) array
            Portfolio weights

        Returns
        -------
        float

        """
        if weights is None:
            nstocks = forecast.shape[0]
            weights = BEKK.weights(nstocks=nstocks)
        else:
            weights = np.array(weights) / np.sum(weights)
        var = BEKK.portf_var(forecast=forecast, alpha=alpha, weights=weights)
        pret = BEKK.pret(innov, weights=weights)
        diff = pret - var
        if diff < 0:
            return 1 + diff ** 2
        else:
            return 0.

    @staticmethod
    def all_losses(forecast=None, proxy=None, innov=None,
                   alpha=.25, kind='equal'):
        """Collect all loss functions.

        Parameters
        ----------
        forecast : (nstocks, nstocks) array
            Volatililty forecast
        proxy : (nstocks, nstocks) array
            Proxy for actual volatility
        innov : (nstocks, ) array
            Returns
        alpha : float
            Risk level. Usually 1% or 5%.
        kind : str
            Either 'equal' or 'minvar' (minimum variance).

        Returns
        -------
        dict

        """
        nstocks = forecast.shape[0]
        weights = BEKK.weights(nstocks=nstocks, hvar=forecast, kind=kind)
        return {'eucl': BEKK.loss_eucl(forecast=forecast, proxy=proxy),
                'frob': BEKK.loss_frob(forecast=forecast, proxy=proxy),
                'stein': BEKK.loss_stein2(forecast=forecast, innov=innov),
                'lsqore': BEKK.portf_lscore(forecast=forecast, innov=innov),
                'mse': BEKK.portf_mse(forecast=forecast, proxy=proxy),
                'qlike': BEKK.portf_qlike(forecast=forecast, proxy=proxy),
                'pret': BEKK.pret(innov, weights=weights),
                'var': BEKK.portf_var(forecast=forecast, alpha=alpha,
                                      weights=weights),
                'var_exception': BEKK.var_exception(innov=innov,
                                                    forecast=forecast,
                                                    alpha=alpha,
                                                    weights=weights),
                'var_loss': BEKK.loss_var(innov=innov, forecast=forecast,
                                          alpha=alpha, weights=weights)}

    @staticmethod
    def collect_losses(param_start=None, innov_all=None, window=1000,
                       model='standard', use_target=False, groups=('NA', 'NA'),
                       restriction='scalar', cfree=False, method='SLSQP',
                       use_penalty=False, ngrid=5, tname='losses'):
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

        groups : tuple
            First item is the string code.
            Second is spatial groups specification.
        use_target : bool
            Whether to use variance targeting (True) or not (False)
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        ngrid : int
            Number of starting values in one dimension
        use_penalty : bool
            Whether to include penalty term in the likelihood
        tname : str
            Name to be used while writing data to the disk

        Returns
        -------
        float
            Average loss_frob function

        """
        nobs = innov_all.shape[0]

        common = {'groups': groups[1], 'use_target': use_target,
                  'model': model, 'restriction': restriction, 'cfree': cfree,
                  'use_penalty': use_penalty}

        loc_name = tname + '_' + model +'_' + restriction + '_' + groups[0]
        fname = '../data/losses/' + loc_name + '.h5'

        for first in range(nobs - window):
            loop = 0
            last = window + first
            innov = innov_all[first:last]
            bekk = BEKK(innov)

            time_start = time.time()
            if first == 0:
                result = bekk.estimate(method=method, **common)
            else:
                result = bekk.estimate(param_start=param_start,
                                       method=method, **common)

            if result.opt_out.fun == 1e10:
                loop = 1
                result = bekk.estimate(param_start=param_start,
                                       method='basin', **common)
            if result.opt_out.fun == 1e10:
                loop = 2
                result = bekk.estimate_loop(ngrid=ngrid, method=method,
                                            **common)

            time_delta = time.time() - time_start

            param_start = result.param_final

            forecast = BEKK.forecast_one(hvar=result.hvar[-1], innov=innov[-1],
                                         param=result.param_final)
            proxy = BEKK.sqinnov(innov_all[last])

            data = BEKK.all_losses(forecast=forecast, proxy=proxy,
                                   innov=innov_all[last])
            data['logl'] = result.opt_out.fun
            data['time_delta'] = time_delta
            data['loop'] = loop

            ids = [model, restriction, groups[0], first]
            names = ['model', 'restriction', 'group', 'first']
            index = pd.MultiIndex.from_arrays(ids, names=names)
            losses = pd.DataFrame(data, index=index)

            append = False if first == 0 else True
            losses.to_hdf(fname, tname, format='t', append=append,
                          min_itemsize=10)

        return pd.read_hdf(fname, tname)
