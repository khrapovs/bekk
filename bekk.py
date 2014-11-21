#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module allows to simulate and estimate the BEKK(1,1) model.

References
----------
.. [1] Robert F. Engle and Kenneth F. Kroner
    "Multivariate Simultaneous Generalized Arch",
    Econometric Theory, Vol. 11, No. 1 (Mar., 1995), pp. 122-150,
    <http://www.jstor.org/stable/3532933>

Notes
-----

Check this repo for related R library: https://github.com/vst/mgarch/

Alternative optimization library: http://www.pyopt.org/

"""
from __future__ import print_function, division

import time

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize

from MGARCH.bekkparams import BEKKParams
from MGARCH.helper_functions import (_bekk_recursion, _product_cc,
	_product_aba, _filter_var, _contribution, estimate_h0, plot_data)

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class BEKK(object):
    r"""BEKK model. Estimation class.

    .. math::
        u_t|H_t &\sim N(0,H_t) \\
        u_t     &= e_t H_t^{1/2}, e_t \sim N(0,I) \\
        H_t     &= E_{t-1}[u_tu_t^\prime]

    One lag, no asymmetries

    .. math::
        H_t = CC^\prime + Au_{t-1}u_{t-1}^\prime A^\prime + BH_{t-1}B^\prime

    Parameters
    ----------
    innov : (nobs, nstocks) array
        Return innovations

    Attributes
    ----------
    innov : (nobs, nstocks) array
        Return innovations
    log_file : str
        File name to write the results of estimation
    param_start : BEKKParams instance
        Initial values of model parameters
    param_final : BEKKParams instance
        Final values of model parameters
    opt_out : scipy.minimize.OptimizeResult instance
        Optimization results

    Methods
    -------
    print_results
        Print results after estimation was completed
    estimate
        Estimate model parameters

    """

    def __init__(self, innov):
        """Initialize the class.

        Parameters
        ----------
        innov : (nobs, nstocks) array
            Return innovations

        """
        self.innov = innov
        self.log_file = 'log.txt'
        self.param_start = None
        self.param_final = None
        self.method = 'L-BFGS-B'
        self.time_delta = None
        self.opt_out = None

    def __likelihood(self, theta, kwargs):
        """Compute the conditional log-likelihood function.

        Parameters
        ----------
        theta : 1dim array
            Dimension depends on the model restriction
        kwargs : dict
            Any additional parameters

        Returns
        -------
        float
            The value of the minus log-likelihood function.
            If some regularity conditions are violated, then it returns
            some obscene number.

        """
        nobs = self.innov.shape[0]
        param = BEKKParams(theta=theta, innov=self.innov,
                           restriction=self.param_start.restriction,
                           var_target=self.param_start.var_target)

        if param.constraint() >= 1:
            return 1e10

        hvar = _filter_var(self.innov, param)

        if not 'parallel' in kwargs:
            kwargs['parallel'] = False

        if not kwargs['parallel']:
            # Serial version
            sumf = 0
            for i in range(nobs):
                fvalue, bad = _contribution(self.innov[i], hvar[i])
                if bad:
                    break
                sumf += fvalue
        else:
            # Parallel version
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(_contribution, zip(self.innov, hvar))
            values, bad = zip(*results)
            sumf = np.array(values).sum()
            bad = np.array(bad).any()

        if np.isinf(sumf) or bad:
            return 1e10
        else:
            return sumf

    def callback(self, theta):
        """Empty callback function.

        Parameters
        ----------
        theta : 1dim array
            Parameter vector

        """
        pass

    def print_results(self, **kwargs):
        """Print stuff after estimation.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments

        """
        if 'param_true' in kwargs:
            like_true = self.__likelihood(kwargs['param_true'].theta, kwargs)
        like_start = self.__likelihood(self.param_start.theta, kwargs)
        like_final = self.__likelihood(self.param_final.theta, kwargs)
        # Form the string
        string = ['\n']
        string.append('Method = ' + self.method)
        string.append('Total time (minutes) = %.2f' % self.time_delta)
        if 'theta_true' in kwargs:
            string.append('True likelihood = %.2f' % like_true)
        string.append('Initial likelihood = %.2f' % like_start)
        string.append('Final likelihood = %.2f' % like_final)
        string.append('Likelihood difference = %.2f' %
                      (like_start - like_final))
        string.append('Success = ' + str(self.opt_out.success))
        string.append('Message = ' + str(self.opt_out.message))
        string.append('Iterations = ' + str(self.opt_out.nit))
        string.extend(self.param_final.log_string())
        string.append('\nH0 target =\n'
                      + np.array_str(estimate_h0(self.innov)))
        # Save results to the log file
        with open(self.log_file, 'a') as texfile:
            for istring in string:
                texfile.write(istring + '\n')

    def estimate(self, restriction='scalar', var_target=True, **kwargs):
        """Estimate parameters of the BEKK model.

        Updates several attributes of the class.

        Parameters
        ----------
        restriction : str
            Can be
                - 'full'
                - 'diagonal'
                - 'scalar'
        var_target : bool
            Variance targeting flag. If True, then c_mat is not returned.
        kwargs : keyword arguments, optional

        """
        # Update default settings
        self.__dict__.update(kwargs)
        # Optimization options
        options = {'disp': False, 'maxiter': int(1e6)}
        # Check for existence of initial guess among arguments.
        # Otherwise, initialize.
        if not 'param_start' in kwargs:
            self.param_start = BEKKParams(restriction=restriction,
                                          var_target=var_target,
                                          innov=self.innov)
        # Start timer for the whole optimization
        time_start = time.time()
        # Run optimization
        self.opt_out = minimize(self.__likelihood,
                                self.param_start.theta,
                                args=(kwargs,),
                                method=self.method,
                                callback=self.callback,
                                options=options)
        # How much time did it take in minutes?
        self.time_delta = (time.time() - time_start) / 60
        # Store optimal parameters in the corresponding class
        self.param_final = BEKKParams(theta=self.opt_out.x,
                                      innov=self.innov,
                                      restriction=restriction,
                                      var_target=var_target)
        if 'log_file' in kwargs:
            self.print_results(**kwargs)

    def estimate_error(self, param):
        """Filter out the error given parameters.

        """
        nobs = self.innov.shape[0]
        hvar = _filter_var(self.innov, param)
        error = np.empty_like(hvar)
        uvar = param.unconditional_var()
        error[0] = _product_cc(self.innov[0]) - uvar
        for i in range(1, nobs):
            error[i] = _product_cc(self.innov[i]) - uvar
            temp = _product_cc(self.innov[i-1]) - uvar
            error[i] -= _product_aba(param.a_mat, temp)
            error[i] -= _product_aba(param.b_mat, hvar[i-1] - uvar)
        return error

    def print_error(self):

        error = self.estimate_error(self.param_final).squeeze()
        plt.plot(error)
        plt.axhline(error.mean())
        plt.show()
        print('Mean error: %.4f' % error.mean())
        print('Estimated H0:', self.param_final.unconditional_var())
        print('Target H0:', estimate_h0(self.innov))


if __name__ == '__main__':
    from usage_example import test_bekk
    test_bekk(nstocks=1, nobs=500, var_target=False)
