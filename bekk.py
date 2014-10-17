#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module allows to simulate and estimate the BEKK(1,1) model.

Robert F. Engle and Kenneth F. Kroner
"Multivariate Simultaneous Generalized Arch"
Econometric Theory, Vol. 11, No. 1 (Mar., 1995), pp. 122-150
http://www.jstor.org/stable/3532933

"""

# Check this repo for related R library: https://github.com/vst/mgarch/
# Alternative optimization library: http://www.pyopt.org/

from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.optimize import minimize
import scipy.linalg as sl
import time

__author__ = "Stanislav Khrapov, Stanislav Anatolyev"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"

class BEKK(object):
    """BEKK model. Estimation class.

    u(t)|H(t) ~ N(0,H(t))
    u(t) = e(t)H(t)^(1/2), e(t) ~ N(0,I)
    H(t) = E_{t-1}[u(t)u(t)']
    One lag, no asymmetries
    H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'

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
        self.restriction = 'scalar'
        self.var_target = False
        self.theta_start = init_parameters(self.innov, self.restriction,
                                           self.var_target)
        # TODO : the following attributes seem excessive:
        self.method = 'L-BFGS-B'
        self.time_start = None
        self.time_final = None
        self.theta_final = None
        self.success = None
        self.nit = None
        self.message = None

    def likelihood(self, theta):
        """Compute the largest eigenvalue of BEKK model.

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
        nobs, nstocks = self.innov.shape
        a_mat, b_mat, c_mat = convert_theta_to_abc(theta, nstocks,
                                                   self.restriction,
                                                   self.var_target)
        if constraint(a_mat, b_mat) >= 1:
            return 1e10

        hvar = filter_var(self.innov, a_mat, b_mat, c_mat, self.var_target)

        sumf = 0
        for tobs in range(nobs):
            fvalue, bad = contribution(self.innov[tobs], hvar[tobs])
            sumf += fvalue
            if bad:
                return 1e10

        if np.isinf(sumf):
            return 1e10
        else:
            return sumf

    def callback(self, theta):
        """Empty callback function."""
        pass

    def print_results(self, **kwargs):
        """Print stuff after estimation.

        """
        nstocks = self.innov.shape[1]
        time_delta = (self.time_final - self.time_start) / 60
        # Convert parameter vector to matrices
        a_mat, b_mat, c_mat = convert_theta_to_abc(self.theta_final, nstocks,
                                                   self.restriction,
                                                   self.var_target)
        if 'theta_true' in kwargs:
            like_true = self.likelihood(kwargs['theta_true'])
        like_start = self.likelihood(self.theta_start)
        like_final = self.likelihood(self.theta_final)
        # Form the string
        string = []
        string.append('Varinace targeting = ' + str(self.var_target))
        string.append('Model restriction = ' + str(self.restriction))
        string.append('Method = ' + self.method)
        string.append('Max eigenvalue = %.4f' % constraint(a_mat, b_mat))
        string.append('Total time (minutes) = %.2f' % time_delta)
        if 'theta_true' in kwargs:
            string.append('True likelihood = %.2f' % like_true)
        string.append('Initial likelihood = %.2f' % like_start)
        string.append('Final likelihood = %.2f' % like_final)
        string.append('Likelihood difference = %.2f' % \
            (like_start - like_final))
        string.append('Success = ' + str(self.success))
        string.append('Message = ' + self.message)
        string.append('Iterations = ' + str(self.nit))
        param_str = ['\nA = ', np.array_str(a_mat),
                     '\nB = ', np.array_str(b_mat)]
        string.extend(param_str)
        if not self.var_target:
            string.extend(['\nC = ', np.array_str(c_mat)])
            stationary_var = find_stationary_var(a_mat, b_mat, c_mat)
            string.extend(['\nH0 estim = ', np.array_str(stationary_var)])
        string.extend(['\nH0 target = ',
                       np.array_str(estimate_h0(self.innov))])

        # Save results to the log file
        with open(self.log_file, 'a') as texfile:
            for istring in string:
                texfile.write(istring + '\n')

    def estimate(self, **kwargs):
        """Estimate parameters of the BEKK model.

        Updates several attributes of the class.

        Parameters
        ----------
        theta0 : 1-dimensional array
            Initial guess. Dimension depends on the problem

        """
        # Update default settings
        self.__dict__.update(kwargs)

        # Start timer for the whole optimization
        self.time_start = time.time()
        # Optimization options
        options = {'disp': False, 'maxiter' : int(1e6)}
        # Run optimization
        output = minimize(self.likelihood, self.theta_start,
                          method=self.method,
                          callback=self.callback,
                          options=options)
        self.theta_final = output.x
        self.success = output.success
        self.nit = output.nit
        self.message = output.message
        # How much time did it take?
        self.time_final = time.time()
        self.print_results(**kwargs)

def simulate_bekk(a_mat, b_mat, c_mat, nobs=1000):
    """Simulate data.

    Parameters
    ----------
    a_mat : (nstocks, nstocks) array
        Parameter matrix
    b_mat : (nstocks, nstocks) array
        Parameter matrix
    c_mat : (nstocks, nstocks) array or None
        Parameter matrix (lower triangular)
    nobs : int
        Number of observations to generate. Time series length

    Returns
    -------
    u: (T, n) array
        multivariate innovation matrix
    """
    nstocks = a_mat.shape[0]
    mean, cov = np.zeros(nstocks), np.eye(nstocks)
    error = np.random.multivariate_normal(mean, cov, nobs)
    hvar = np.empty((nobs, nstocks, nstocks))
    innov = np.zeros((nobs, nstocks))

    hvar[0] = find_stationary_var(a_mat, b_mat, c_mat)

    for tobs in range(1, nobs):
        hvar[tobs] = c_mat.dot(c_mat.T)
        innov2 = innov[tobs-1, np.newaxis].T * innov[tobs-1]
        hvar[tobs] += a_mat.dot(innov2).dot(a_mat.T)
        hvar[tobs] += b_mat.dot(hvar[tobs-1]).dot(b_mat.T)
        hvar12 = sp.linalg.cholesky(hvar[tobs], 1)
        innov[tobs] = hvar12.dot(np.atleast_2d(error[tobs]).T).flatten()

    return innov

def filter_var(innov, a_mat, b_mat, c_mat, var_target):
    """Filter out variances and covariances of innovations.

    Parameters
    ----------
    innov : (nobs, nstocks) array
        Return innovations
    a_mat : (nstocks, nstocks) array
        Parameter matrix
    b_mat : (nstocks, nstocks) array
        Parameter matrix
    c_mat : (nstocks, nstocks) array or None
        Parameter matrix (lower triangular)
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
        stationary_var = find_stationary_var(a_mat, b_mat, c_mat)

    hvar = np.empty((nobs, nstocks, nstocks))
    hvar[0] = stationary_var.copy()

    for tobs in range(1, nobs):
        hvar[tobs] = hvar[0]
        innov2 = innov[tobs-1, np.newaxis].T * innov[tobs-1]
        hvar[tobs] += a_mat.dot(innov2 - hvar[0]).dot(a_mat.T)
        hvar[tobs] += b_mat.dot(hvar[tobs-1] - hvar[0]).dot(b_mat.T)

    return hvar

#@profile
def contribution(innov, hvar):
    """Contribution to the log-likelihood function for each observation.

    Parameters
    ----------
    innov: (n,) array
        inovations
    hvar: (n, n) array
        variance/covariances

    Returns
    -------
    f: float
        log-likelihood contribution
    bad: bool
        True if something is wrong
    """

    try:
        lu_decomp, piv = sl.lu_factor(hvar)
    except (sl.LinAlgError, ValueError):
        return 1e10, True

    hvardet = np.abs(np.prod(np.diag(lu_decomp)))
    if hvardet > 1e20 or hvardet < 1e-5:
        return 1e10, True

    norm_innov = sl.lu_solve((lu_decomp, piv), innov)
    fvalue = np.log(hvardet) + norm_innov.dot(np.atleast_2d(innov).T)
    if np.isinf(fvalue):
        return 1e10, True
    else:
        return float(fvalue), False

def estimate_h0(innov):
    """Estimate unconditional realized covariance matrix.

    Parameters
    ----------
    innov: (T, n) array
        inovations

    Returns
    -------
    (n, n) array
        E[innov' * innov]

    """
    return innov.T.dot(innov) / innov.shape[0]

def convert_theta_to_abc(theta, nstocks,
                         restriction='scalar', var_target=True):
    """Convert 1-dimensional array of parameters to matrices.

    Parameters
    ----------
    theta: 1d array of parameters
        Length depends on the model restrictions and variance targeting
        If var_targeting:
            'full' - 2*n**2
            'diagonal' - 2*n
            'scalar' - 2
        If not var_targeting:
            + (n-1)*n/2 for parameter C
    nstocks: number of innovations in the model

    Returns
    -------
    a_mat : (nstocks, nstocks) array
        Parameter matrix
    b_mat : (nstocks, nstocks) array
        Parameter matrix
    c_mat : (nstocks, nstocks) array or None
        Parameter matrix (lower triangular)
    restriction : str
        Can be 'full', 'diagonal', 'scalar'
    var_target : bool
        Variance targeting flag. If True, then c_mat is not returned.

    """
    if restriction == 'full':
        chunk = nstocks**2
        a_mat = theta[:chunk].reshape([nstocks, nstocks])
        b_mat = theta[chunk:2*chunk].reshape([nstocks, nstocks])
    elif restriction == 'diagonal':
        chunk = nstocks
        a_mat = np.diag(theta[:chunk])
        b_mat = np.diag(theta[chunk:2*chunk])
    elif restriction == 'scalar':
        chunk = 1
        a_mat = np.eye(nstocks) * theta[:chunk]
        b_mat = np.eye(nstocks) * theta[chunk:2*chunk]
    else:
        raise ValueError('This restriction is not supported!')

    if var_target:
        c_mat = None
    else:
        c_mat = np.zeros((nstocks, nstocks))
        c_mat[np.tril_indices(nstocks)] = theta[2*chunk:]
    return a_mat, b_mat, c_mat

def convert_abc_to_theta(a_mat, b_mat, c_mat,
                         restriction='scalar', var_target=True):
    """Convert parameter matrices to 1-dimensional array.

    Parameters
    ----------
    a_mat : (nstocks, nstocks) array
        Parameter matrix
    b_mat : (nstocks, nstocks) array
        Parameter matrix
    c_mat : (nstocks, nstocks) array
        Parameter matrix (lower triangular)
    restriction : str
        Model type. Can be 'full', 'diagonal', 'scalar'
    var_target : bool
        Variance targeting flag. If True, then c_mat is not returned.

    Returns
    -------
    1-dimensional array of parameters
        Length depends on the model restrictions and variance targeting
        If var_targeting:
            'full' - 2*n**2
            'diagonal' - 2*n
            'scalar' - 2
        If not var_targeting:
            + (n-1)*n/2 for parameter c_mat

    """
    if restriction == 'full':
        theta = [a_mat.flatten(), b_mat.flatten()]
    elif restriction == 'diagonal':
        theta = [np.diag(a_mat), np.diag(b_mat)]
    elif restriction == 'scalar':
        theta = [[a_mat[0, 0]], [b_mat[0, 0]]]
    else:
        raise ValueError('This restriction is not supported!')
    if not var_target:
        theta.append(c_mat[np.tril_indices(c_mat.shape[0])])
    return np.concatenate(theta)

def find_stationary_var(a_mat, b_mat, c_mat):
    """Find fixed point of H = CC' + AHA' + BHB'.

    Parameters
    ----------
    a_mat, b_mat, c_mat: (n, n) arrays
        Parameter matrices

    Returns
    -------
        (n, n) array
    """
    i, norm = 0, 1e3
    hvarold = np.eye(a_mat.shape[0])
    while (norm > 1e-3) or (i < 1000):
        hvarnew = c_mat.dot(c_mat.T) + \
            a_mat.dot(hvarold).dot(a_mat.T) + b_mat.dot(hvarold).dot(b_mat.T)
        norm = np.linalg.norm(hvarnew - hvarold)
        hvarold = hvarnew[:]
        i += 1
    return hvarnew

def find_c_mat(a_mat, b_mat, stationary_var):
    """Find C in H = CC' + AHA' + BHB'.

    Parameters
    ----------
    a_mat, b_mat, stationary_var: (n, n) arrays
        Parameter matrices

    Returns
    -------
    c_mat : (n, n) array
        Lower triangular matrix
    """
    c_mat_sq = stationary_var - a_mat.dot(stationary_var).dot(a_mat.T) \
        - b_mat.dot(stationary_var).dot(b_mat.T)
    # Extract C parameter
    return sp.linalg.cholesky(c_mat_sq, 1)

def init_parameters(innov, restriction, var_target):
    """Initialize parameters for further estimation.

    Parameters
    ----------
    innov: (T, n) array
        Return innovations
    restriction : str
        Model type. Can be 'full', 'diagonal', 'scalar'
    var_target : bool
        Variance targeting flag. If True, then c_mat is not returned.

    Returns
    -------
    theta : 1-dimensional array of parameters
        The initial guess for parameters.
        Length depends of restriction and var_target.

    """
    nstocks = innov.shape[1]
    a_mat = np.eye(nstocks) * .15
    b_mat = np.eye(nstocks) * .95
    # Estimate stationary variance
    stationary_var = estimate_h0(innov)
    # Compute the constant term
    c_mat = find_c_mat(a_mat, b_mat, stationary_var)
    theta = convert_abc_to_theta(a_mat, b_mat, c_mat, restriction, var_target)
    return theta

def constraint(a_mat, b_mat):
    """Compute the largest eigenvalue of BEKK model.

    Parameters
    ----------
    a_mat : (n, n) array
    b_mat : (n, n) array

    Returns
    -------
    float

    """
    return np.abs(sl.eigvals(np.kron(a_mat, a_mat) \
        + np.kron(b_mat, b_mat))).max()

def plot_data(innov, hvar):
    """Plot time series of hvar and u elements.

    Parameters
    ----------
    innov: (T, n) array
        innovations
    hvar: (T, n, n) array
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
    test_simulate(n=2, T=500)
