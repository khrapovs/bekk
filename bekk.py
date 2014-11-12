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
import numpy as np
import scipy.linalg as sl
import multiprocessing as mp
from scipy.optimize import minimize
from functools import reduce


__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class BEKKParams(object):
    """Class to hold parameters of the BEKK model in different representations.

    Attributes
    ----------
    a_mat, b_mat, c_mat : (nstocks, nstocks) arrays
        Matrix representations of BEKK parameters
    theta : 1-dimensional array
        Vector of model parameters
    nstocks : int
        Number of innovations in the model
    restriction : str
        Can be
            - 'full'
            - 'diagonal'
            - 'scalar'
    var_target : bool
        Variance targeting flag. If True, then c_mat is not returned.

    """

    def __init__(self, restriction=None, var_target=None, **kwargs):
        """Class constructor.

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
        # Defaults:
        self.a_mat, self.b_mat, self.c_mat = None, None, None
        self.theta = None
        self.nstocks = None
        self.restriction = restriction
        self.var_target = var_target
        # Update attributes from kwargs
        self.__dict__.update(kwargs)

        if 'theta' in kwargs:
            self.__convert_theta_to_abc()
        elif 'a_mat' and 'b_mat' in kwargs:
            self.__convert_abc_to_theta()
        elif 'innov' in kwargs:
            self.__init_parameters(kwargs['innov'])
        else:
            raise TypeError('Not enough arguments to initialize BEKKParams!')

    def __init_parameters(self, innov):
        """Initialize parameter class given innovations only.

        Parameters
        ----------
        innov: (nobs, ntocks) array
            Return innovations

        """
        self.nstocks = innov.shape[1]
        self.a_mat = np.eye(self.nstocks) * .15
        self.b_mat = np.eye(self.nstocks) * .95
        # Estimate stationary variance
        stationary_var = estimate_h0(innov)
        # Compute the constant term
        self.__find_c_mat(stationary_var)
        self.__convert_abc_to_theta()

    def __find_c_mat(self, stationary_var):
        """Solve for C in H = CC' + AHA' + BHB' given A, B, H.

        Parameters
        ----------
        stationary_var: (nstocks, nstocks) arrays
            Stationary variance matrix, H

        """
        c_mat_sq = stationary_var \
            - _product_aba(self.a_mat, stationary_var) \
            - _product_aba(self.b_mat, stationary_var)
        # Extract C parameter
        self.c_mat = sl.cholesky(c_mat_sq, 1)

    def __convert_theta_to_abc(self):
        """Convert 1-dimensional array of parameters to matrices.

        Notes
        -----
        a_mat, b_mat, c_mat : (nstocks, nstocks) arrays
            Parameter matrices
        theta: 1d array of parameters
            Length depends on the model restrictions and variance targeting
            If var_targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not var_targeting:
                - + (n-1)*n/2 for parameter C

        """
        if self.restriction == 'full':
            chunk = self.nstocks**2
            sqsize = [self.nstocks, self.nstocks]
            self.a_mat = self.theta[:chunk].reshape(sqsize)
            self.b_mat = self.theta[chunk:2*chunk].reshape(sqsize)
        elif self.restriction == 'diagonal':
            chunk = self.nstocks
            self.a_mat = np.diag(self.theta[:chunk])
            self.b_mat = np.diag(self.theta[chunk:2*chunk])
        elif self.restriction == 'scalar':
            chunk = 1
            self.a_mat = np.eye(self.nstocks) * self.theta[:chunk]
            self.b_mat = np.eye(self.nstocks) * self.theta[chunk:2*chunk]
        else:
            raise ValueError('This restriction is not supported!')

        if self.var_target:
            self.c_mat = None
        else:
            self.c_mat = np.zeros((self.nstocks, self.nstocks))
            self.c_mat[np.tril_indices(self.nstocks)] = self.theta[2*chunk:]

    def __convert_abc_to_theta(self):
        """Convert parameter matrices to 1-dimensional array.

        Notes
        -----
        a_mat, b_mat, c_mat : (nstocks, nstocks) arrays
            Parameter matrices
        theta : 1-dimensional array of parameters
            Length depends on the model restrictions and variance targeting
            If var_targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not var_targeting:
                - + (n-1)*n/2 for parameter c_mat

        """
        if self.restriction == 'full':
            self.theta = [self.a_mat.flatten(), self.b_mat.flatten()]
        elif self.restriction == 'diagonal':
            self.theta = [np.diag(self.a_mat), np.diag(self.b_mat)]
        elif self.restriction == 'scalar':
            self.theta = [[self.a_mat[0, 0]], [self.b_mat[0, 0]]]
        else:
            raise ValueError('This restriction is not supported!')
        if not self.var_target:
            self.theta.append(self.c_mat[np.tril_indices(self.c_mat.shape[0])])
        self.theta = np.concatenate(self.theta)

    def find_stationary_var(self):
        """Find fixed point of H = CC' + AHA' + BHB' given A, B, C.

        Returns
        -------
        hvarnew : (nstocks, nstocks) array
            Stationary variance amtrix

        """
        i, norm = 0, 1e3
        hvarold = np.eye(self.a_mat.shape[0])
        while (norm > 1e-3) or (i < 100):
            hvarnew = _product_cc(self.c_mat) \
                + _product_aba(self.a_mat, hvarold) \
                + _product_aba(self.b_mat, hvarold)
            norm = np.linalg.norm(hvarnew - hvarold)
            hvarold = hvarnew[:]
            i += 1
        return hvarnew

    def constraint(self):
        """Compute the largest eigenvalue of BEKK model.

        Returns
        -------
        float
            Largest eigenvalue

        """
        kron_a = np.kron(self.a_mat, self.a_mat)
        kron_b = np.kron(self.b_mat, self.b_mat)
        return np.abs(sl.eigvals(kron_a + kron_b)).max()

    def log_string(self):
        """Create string for log file.

        Returns
        -------
        string : list
            List of strings

        """
        string = []
        string.append('Varinace targeting = ' + str(self.var_target))
        string.append('Model restriction = ' + str(self.restriction))
        string.append('Max eigenvalue = %.4f' % self.constraint())
        string.append('\nA =\n' + np.array_str(self.a_mat))
        string.append('\nB =\n' + np.array_str(self.b_mat))
        if self.c_mat is not None:
            string.append('\nC =\n' + np.array_str(self.c_mat))
            string.append('\nH0 estim =\n'
                          + np.array_str(self.find_stationary_var()))
        return string


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
        nobs, nstocks = self.innov.shape
        param = BEKKParams(theta=theta, nstocks=nstocks,
                           restriction=self.param_start.restriction,
                           var_target=self.param_start.var_target)

        if param.constraint() >= 1:
            return 1e10

        hvar = _filter_var(self.innov, param, self.param_start.var_target)

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
                                      nstocks=self.innov.shape[1],
                                      restriction=restriction,
                                      var_target=var_target)
        if 'log_file' in kwargs:
            self.print_results(**kwargs)


def simulate_bekk(param, nobs=1000):
    """Simulate data.

    Parameters
    ----------
    param : instance of BEKKParams class
        Attributes of this class hold parameter matrices
    nobs : int, optional
        Number of observations to generate. Time series length

    Returns
    -------
    innov : (nobs, nstocks) array
        Multivariate innovation matrix

    """
    nstocks = param.a_mat.shape[0]
    mean, cov = np.zeros(nstocks), np.eye(nstocks)
    error = np.random.multivariate_normal(mean, cov, nobs)
    hvar = np.empty((nobs, nstocks, nstocks))
    innov = np.zeros((nobs, nstocks))

    hvar[0] = param.find_stationary_var()

    for i in range(1, nobs):
        hvar[i] = _product_cc(param.c_mat)
        innov2 = innov[i-1, np.newaxis].T * innov[i-1]
        hvar[i] += _product_aba(param.a_mat, innov2)
        hvar[i] += _product_aba(param.b_mat, hvar[i-1])
        hvar12 = sl.cholesky(hvar[i], 1)
        innov[i] = hvar12.dot(np.atleast_2d(error[i]).T).flatten()

    return innov


def _product_cc(mat):
    """Compute CC'.

    Parameters
    ----------
    mat : 2dim square array

    Returns
    -------
    mat : 2dim square array

    """
    return mat.dot(mat.T)


def _product_aba(a_mat, b_mat):
    """Compute ABA'.

    Parameters
    ----------
    a_mat, b_mat : 2dim arrays

    Returns
    -------
    mat : 2dim square array

    """
    return reduce(np.dot, [a_mat, b_mat, a_mat.T])


def _filter_var(innov, param, var_target):
    """Filter out variances and covariances of innovations.

    Parameters
    ----------
    innov : (nobs, nstocks) array
        Return innovations
    param : instance of BEKKParams class
        Attributes of this class hold parameter matrices
    var_target : bool
        Variance targeting flag

    Returns
    -------
    hvar : (nobs, nstocks, nstocks) array
        Variances and covariances of innovations

    """

    nobs, nstocks = innov.shape

    if var_target:
        # Estimate unconditional realized covariance matrix
        stationary_var = estimate_h0(innov)
    else:
        stationary_var = param.find_stationary_var()

    hvar = np.empty((nobs, nstocks, nstocks))
    hvar[0] = stationary_var.copy()

    for i in range(1, nobs):
        hvar[i] = hvar[0]
        innov2 = innov[i-1, np.newaxis].T * innov[i-1] - hvar[0]
        hvar[i] += _product_aba(param.a_mat, innov2)
        hvar[i] += _product_aba(param.b_mat, hvar[i-1]-hvar[0])

    return hvar


def _contribution(innov, hvar):
    """Contribution to the log-likelihood function for each observation.

    Parameters
    ----------
    innov: (nstocks,) array
        inovations
    hvar: (nstocks, nstocks) array
        variance/covariances

    Returns
    -------
    fvalue : float
        log-likelihood contribution
    bad : bool
        True if something is wrong
    """

    lower = True
    try:
        cho_c = sl.cholesky(hvar, lower=lower)
    except (sl.LinAlgError, ValueError):
        return 1e10, True

    hvardet = sl.det(cho_c)**2
    norm_innov = sl.cho_solve((cho_c, lower), innov)
    fvalue = np.log(hvardet) + (norm_innov * innov).sum()

    if np.isinf(fvalue):
        return 1e10, True
    else:
        return float(fvalue), False


def estimate_h0(innov):
    """Estimate unconditional realized covariance matrix.

    Parameters
    ----------
    innov: (nobs, nstocks) array
        inovations

    Returns
    -------
    (nstocks, nstocks) array
        E[innov' * innov]

    """
    return innov.T.dot(innov) / innov.shape[0]


def plot_data(innov, hvar):
    """Plot time series of hvar and u elements.

    Parameters
    ----------
    innov: (nobs, nstocks) array
        innovations
    hvar: (nobs, nstocks, nstocks) array
        variance/covariances
    """
    nobs, nstocks = innov.shape
    axes = plt.subplots(nrows=nstocks**2, ncols=1)[1]
    for axi, i in zip(axes, range(nstocks**2)):
        axi.plot(range(nobs), hvar.reshape([nobs, nstocks**2])[:, i])
    plt.plot()

    axes = plt.subplots(nrows=nstocks, ncols=1)[1]
    for axi, i in zip(axes, range(nstocks)):
        axi.plot(range(nobs), innov[:, i])
    plt.plot()


if __name__ == '__main__':
    from MGARCH.usage_example import test_simulate
    test_simulate(nstocks=2, nobs=500)
