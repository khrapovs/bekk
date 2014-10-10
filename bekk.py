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
        self.nobs, self.nstocks = innov.shape

    def constraint(self, A, B):
        """Compute the largest eigenvalue of BEKK model.

        Parameters
        ----------
        A : (n, n) array
        B : (n, n) array

        Returns
        -------
        float

        """
        return np.abs(sl.eigvals(np.kron(A, A) + np.kron(B, B))).max()

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
        A, B = convert_theta_to_abc(theta, self.nstocks,
                                    self.restriction, self.var_target)
        if self.constraint(A, B) >= 1:
            return 1e10
        H = np.empty((self.nobs, self.nstocks, self.nstocks))
        
        # Estimate unconditional realized covariance matrix
        H[0] = estimate_H0(self.innov)

        for t in range(1, self.nobs):
            H[t] = H[0]
            uu = self.innov[t-1, np.newaxis].T * self.innov[t-1]
            H[t] += A.dot(uu - H[0]).dot(A.T)
            H[t] += B.dot(H[t-1] - H[0]).dot(B.T)
        self.H = H
        sumf = 0
        for t in range(self.nobs):
            f, bad = contribution(self.innov[t], self.H[t])
            sumf += f
            if bad:
                return 1e10

        if np.isinf(sumf):
            return 1e10
        else:
            return sumf

    def callback(self, xk):
        """Print stuff for each iteraion.

        Parameters
        ----------
        xk: 1-dimensional array
            Current parameter value. Dimension depends on the problem
        """
        self.iteration += 1
        A, B = convert_theta_to_abc(xk, self.nstocks,
                                    self.restriction, self.var_target)

        start_like = self.likelihood(self.theta_start)
        current_like = self.likelihood(xk)
        true_like = 0
        old_like = self.likelihood(self.xk_old)

        time_new = time.time()
        time_diff = (time_new - self.time_old) / 60
        since_start = (time_new - self.time_start) / 60

        self.xk_old = xk.copy()
        self.time_old = time_new

        string = ['\nIteration = ' + str(self.iteration)]
        string.append('Time spent (minutes) = %.2f' % time_diff)
        string.append('Since start (minutes) = %.2f' % since_start)
        string.append('Initial likelihood = %.2f' % start_like)
        string.append('Current likelihood = %.2f' % current_like)
        string.append('Current - true likelihood = %.2f' \
            % (current_like - true_like))
        string.append('Current - previous likelihood = %.2f' \
            % (current_like - old_like))
        string.extend(['A = ', np.array_str(A), 'B = ', np.array_str(B)])
        with open(self.log_file, 'a') as texfile:
            for s in string:
                texfile.write(s + '\n')

    def print_results(self, **kwargs):
        """Print stuff after estimation.

        """
        self.theta_final = self.res.x
        time_delta = (self.time_final - self.time_start) / 60
        # Convert parameter vector to matrices
        A, B = convert_theta_to_abc(self.theta_final, self.nstocks,
                                    self.restriction, self.var_target)
        if kwargs['theta_true'] is not None:
            like_true = self.likelihood(kwargs['theta_true'])
        like_start = self.likelihood(self.theta_start)
        like_final = self.likelihood(self.theta_final)
        like_delta = like_start - like_final
        # Save results to the log file
        string = ['\n\nMethod : ' + self.method]
        string.append('Max eigenvalue = %.4f' % self.constraint(A, B))
        string.append('Total time (minutes) = %.2f' % time_delta)
        if kwargs['theta_true'] is not None:
            string.append('True likelihood = %.2f' % like_true)
        string.append('Initial likelihood = %.2f' % like_start)
        string.append('Final likelihood = %.2f' % like_final)
        string.append('Likelihood difference = %.2f' % like_delta)
        string.append(str(self.res))
        string.extend(['A = ', np.array_str(A), 'B = ', np.array_str(B)])
        with open(self.log_file, 'a') as texfile:
            for s in string:
                texfile.write(s + '\n')

    def estimate(self, **kwargs):
        """Estimate parameters of the BEKK model.

        Updates several attributes of the class.

        Parameters
        ----------
        theta0: 1-dimensional array
            Initial guess. Dimension depends on the problem

        """
        if not 'var_target' in kwargs:
            self.var_target = True
        else:
            self.var_target = kwargs['var_target']
        if not 'restriction' in kwargs:
            self.restriction = 'scalar'
        else:
            self.restriction = kwargs['theta_start']
        if not 'theta_start' in kwargs:
            self.theta_start = init_parameters(self.restriction, self.nstocks)
        else:
            self.theta_start = kwargs['theta_start']
        if not 'log_file' in kwargs:
            self.log_file = 'log.txt'
        else:
            self.log_file = kwargs['log_file']
        if not 'method' in kwargs:
            self.method = 'L-BFGS-B'
        else:
            self.method = kwargs['method']
        if not 'maxiter' in kwargs:
            maxiter = 1e6
        else:
            maxiter = kwargs['maxiter']
        if not 'use_callback' in kwargs:
            use_callback = False
        else:
            use_callback = kwargs['use_callback']
        if use_callback:
            callback = self.callback
        else:
            callback = None

        self.xk_old = self.theta_start.copy()
        # Iteration number
        self.iteration = 0
        # Start timer for the whole optimization
        self.time_start = time.time()
        self.time_old = time.time()
        # Optimization options
        options = {'disp': False, 'maxiter' : int(maxiter)}
        # Run optimization
        self.res = minimize(self.likelihood, self.theta_start,
                            method=self.method,
                            callback=callback,
                            options=options)
        self.theta_final = self.res.x
        # How much time did it take?
        self.time_final = time.time()
        self.print_results(**kwargs)

def simulate_BEKK(A, B, C, T=1000):
    """Simulate data.

    Parameters
    ----------
    theta : 1-dim array
        True model parameters.
    n : int
        Number of series to simulate
    T : int
        Number of observations to generate. Time series length

    Returns
    -------
    u: (T, n) array
        multivariate innovation matrix
    """
    n = A.shape[0]
    mean, cov = np.zeros(n), np.eye(n)
    e = np.random.multivariate_normal(mean, cov, T)
    H = np.empty((T, n, n))
    innov = np.zeros((T, n))

    H[0] = find_stationary_var(A, B, C)

    for t in range(1, T):
        H[t] = C.dot(C.T)
        H[t] += A.dot(innov[t-1, np.newaxis].T * innov[t-1]).dot(A.T)
        H[t] += B.dot(H[t-1]).dot(B.T)
        H12 = sp.linalg.cholesky(H[t], 1)
        innov[t] = H12.dot(np.atleast_2d(e[t]).T).flatten()

    return innov

#@profile
def contribution(innov, H):
    """Contribution to the log-likelihood function for each observation.

    Parameters
    ----------
    innov: (n,) array
        inovations
    H: (n, n) array
        variance/covariances

    Returns
    -------
    f: float
        log-likelihood contribution
    bad: bool
        True if something is wrong
    """

    try:
        LU, piv = sl.lu_factor(H)
    except (sl.LinAlgError, ValueError):
        return 1e10, True

    Hdet = np.abs(np.prod(np.diag(LU)))
    if Hdet > 1e20 or Hdet < 1e-5:
        return 1e10, True

    y = sl.lu_solve((LU, piv), innov)
    f = np.log(Hdet) + y.dot(np.atleast_2d(innov).T)
    if np.isinf(f):
        return 1e10, True
    else:
        return float(f), False


def estimate_H0(innov):
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
    """Convert 1-dimensional array of parameters to matrices A, B, and C.

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
    A : (nstocks, nstocks) array
        Parameter matrix
    B : (nstocks, nstocks) array
        Parameter matrix
    C : (nstocks, nstocks) array
        Parameter matrix (lower triangular)
    restriction : str
        Can be 'full', 'diagonal', 'scalar'
    var_target : bool
        Variance targeting flag. If True, then C is not returned.
        
    """
    if restriction == 'full':
        chunk = nstocks**2
        A = theta[:chunk].reshape([nstocks, nstocks])
        B = theta[chunk:2*chunk].reshape([nstocks, nstocks])
    elif restriction == 'diagonal':
        chunk = nstocks
        A = np.diag(theta[:chunk])
        B = np.diag(theta[chunk:2*chunk])
    elif restriction == 'scalar':
        chunk = 1
        A = np.eye(nstocks) * theta[:chunk]
        B = np.eye(nstocks) * theta[chunk:2*chunk]
    else:
        raise ValueError('This restriction is not supported!')
    
    if var_target:
        return A, B
    else:
        C = np.zeros((nstocks, nstocks))
        C[np.tril_indices(nstocks)] = theta[2*chunk:]
        return A, B, C

def convert_abc_to_theta(A, B, C, restriction='scalar', var_target=True):
    """Convert parameter matrices A, B, and C to 1-dimensional array.

    Parameters
    ----------
    A : (nstocks, nstocks) array
        Parameter matrix
    B : (nstocks, nstocks) array
        Parameter matrix
    C : (nstocks, nstocks) array
        Parameter matrix (lower triangular)
    restriction : str
        Can be 'full', 'diagonal', 'scalar'
    var_target : bool
        Variance targeting flag. If True, then C is not returned.

    Returns
    -------
    1-dimensional array of parameters
        Length depends on the model restrictions and variance targeting
        If var_targeting:
            'full' - 2*n**2
            'diagonal' - 2*n
            'scalar' - 2
        If not var_targeting:
            + (n-1)*n/2 for parameter C
            
    """
    if restriction == 'full':
        theta = [A.flatten(), B.flatten()]
    elif restriction == 'diagonal':
        theta = [np.diag(A), np.diag(B)]
    elif restriction == 'scalar':
        return np.array([A[0, 0], B[0, 0]])
    else:
        raise ValueError('This restriction is not supported!')
    if var_target:
        theta.append(C[np.tril_indices(C.shape[0])])
    return np.concatenate(theta)

def find_stationary_var(A, B, C):
    """Find fixed point of H = CC' + AHA' + BHB'.

    Parameters
    ----------
    A, B, C: (n, n) arrays
        Parameter matrices

    Returns
    -------
        (n, n) array
    """
    i, norm = 0, 1e3
    Hold = np.eye(A.shape[0])
    while (norm > 1e-3) or (i < 1000):
        Hnew = C.dot(C.T) + A.dot(Hold).dot(A.T) + B.dot(Hold).dot(B.T)
        norm = np.linalg.norm(Hnew - Hold)
        Hold = Hnew[:]
        i += 1
    return Hnew

def init_parameters(innov, restriction, nstocks):
    """Initialize parameters for further estimation.
    
    Parameters
    ----------
        restriction : str
            Type of the model choisen from ['scalar', 'diagonal', 'full']
        n : int
            Number of assets in the model.
    
    Returns
    -------
        theta : (n,) array
            The initial guess for parameters.
    """
    nstocks = innov.shape
    A = np.eye(nstocks) * .15 # + np.ones((n, n)) *.05
    B = np.eye(nstocks) * .95
    # Estimate stationary variance
    stationary_var = estimate_H0(innov)
    # Compute the constant term
    CC = stationary_var - A.dot(stationary_var).dot(A.T) \
        - B.dot(stationary_var).dot(B.T)
    # Extract C parameter
    C = sp.linalg.cholesky(CC, 1)
    theta = convert_abc_to_theta(A, B, C, restriction, var_target)
    return theta

def plot_data(innov, H):
    """Plot time series of H and u elements.

    Parameters
    ----------
    innov: (T, n) array
        innovations
    H: (T, n, n) array
        variance/covariances
    """
    T, n = innov.shape
    fig, axes = plt.subplots(nrows=n**2, ncols=1)
    for ax, i in zip(axes, range(n**2)):
        ax.plot(range(T), H.reshape([T, n**2])[:, i])
    plt.plot()

    fig, axes = plt.subplots(nrows=n, ncols=1)
    for ax, i in zip(axes, range(n)):
        ax.plot(range(T), innov[:, i])
    plt.plot()

if __name__ == '__main__':
    from MGARCH.usage_example import test_simulate
    test_simulate(n=2, T=100)
