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

    def __init__(self, u):
        # Vector of innovations, T x n
        self.u = u
        self.T, self.n = u.shape
        self.theta_true = None
        # Estimate unconditional realized covariance matrix
        self.H0 = estimate_H0(u)
        self.method = 'L-BFGS-B'
        self.use_callback = False
        self.restriction = 'scalar'
    
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
        A, B = convert_theta_to_ab(theta, self.n, self.restriction)
        if self.constraint(A, B) >= 1:
            return 1e10
        H = np.empty((self.T, self.n, self.n))
        
        H[0] = self.H0
        
        for t in range(1, self.T):
            H[t] = H[0]
            uu = self.u[t-1, np.newaxis].T * self.u[t-1]
            H[t] += A.dot(uu - H[0]).dot(A.T)
            H[t] += B.dot(H[t-1] - H[0]).dot(B.T)
        self.H = H
        sumf = 0
        for t in range(self.T):
            f, bad = contribution(self.u[t], self.H[t])
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
        self.it += 1
        A, B = convert_theta_to_ab(xk, self.n, self.restriction)
        
        start_like = self.likelihood(self.theta_start)
        current_like = self.likelihood(xk)
        true_like = 0
        if self.theta_true is not None:
            true_like = self.likelihood(self.theta_true)
        old_like = self.likelihood(self.xk_old)
        
        time_new = time.time()
        time_diff = (time_new - self.time_old) / 60
        since_start = (time_new - self.time_start) / 60

        self.xk_old = xk.copy()
        self.time_old = time_new
        
        string = ['\nIteration = ' + str(self.it)]
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
    
    def print_results(self):
        """Print stuff after estimation.
        
        """
        self.theta_final = self.res.x
        time_delta = (self.time_final - self.time_start) / 60
        # Convert parameter vector to matrices
        A, B = convert_theta_to_ab(self.theta_final, self.n, self.restriction)
        like_start = self.likelihood(self.theta_start)
        like_final = self.likelihood(self.theta_final)
        like_delta = like_start - like_final
        # Save results to the log file
        string = ['\n\nMethod : ' + self.method]
        string.append('Max eigenvalue = %.4f' % self.constraint(A, B))
        string.append('Total time (minutes) = %.2f' % time_delta)
        string.append('Initial likelihood = %.2f' % like_start)
        string.append('Final likelihood = %.2f' % like_final)
        string.append('Likelihood difference = %.2f' % like_delta)
        string.append(str(self.res))
        string.extend(['A = ', np.array_str(A), 'B = ', np.array_str(B)])
        with open(self.log_file, 'a') as texfile:
            for s in string:
                texfile.write(s + '\n')
    
    def estimate(self, theta0):
        """Estimate parameters of the BEKK model.
        
        Updates several attributes of the class.
        
        Parameters
        ----------
            theta0: 1-dimensional array
                Initial guess. Dimension depends on the problem
        """
        self.theta_start = theta0
        self.xk_old = theta0
        self.it = 0
        # Start timer for the whole optimization
        self.time_start = time.time()
        self.time_old = time.time()
        if self.use_callback:
            callback = self.callback
        else:
            callback = None
        # Optimization options
        options = {'disp': False, 'maxiter' : int(self.maxiter)}
        # Run optimization
        self.res = minimize(self.likelihood, self.theta_start,
                       method = self.method,
                       callback = callback,
                       options = options)
        self.theta_final = self.res.x
        # How much time did it take?
        self.time_final = time.time()
        self.print_results()
        
def simulate_BEKK(theta0, n = 2, T = 1000, log = 'bekk_log.txt'):
    """Simulate data.
    
    Parameters
    ----------
        theta0 : 1-dim array
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
    
    A, B, C = convert_theta_to_abc(theta0, n)
    mean, cov = np.zeros(n), np.eye(n)
    
    e = np.random.multivariate_normal(mean, cov, T)
    H = np.empty((T, n, n))
    u = np.zeros((T, n))
    
    H[0] = stationary_H(A, B, C)
    
    for t in range(1, T):
        H[t] = C.dot(C.T)
        H[t] += A.dot(u[t-1, np.newaxis].T * u[t-1]).dot(A.T)
        H[t] += B.dot(H[t-1]).dot(B.T)
        H12 = sp.linalg.cholesky(H[t], 1)
        u[t] = H12.dot(np.atleast_2d(e[t]).T).flatten()
    
    return u

#@profile
def contribution(u, H):
    """Contribution to the log-likelihood function for each observation.
    
    Parameters
    ----------
        u: (n,) array
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
    
    y = sl.lu_solve((LU, piv), u)
    f = np.log(Hdet) + y.dot(np.atleast_2d(u).T)
    if np.isinf(f):
        return 1e10, True
    else:
        return float(f), False
    
    
def estimate_H0(u):
    """Estimate unconditional realized covariance matrix.
    
    Parameters
    ----------
        u: (T, n) array
            inovations
    
    Returns
    -------
        (n, n) array
            E[u'u]
    """
    T = u.shape[0]
    return u.T.dot(u) / T

def convert_theta_to_abc(theta, n):
    """Convert 1-dimensional array of parameters to matrices A, B, and C.
    
    Parameters
    ----------
        theta: array of parameters
            Length depends on the model restrictions:
            'full' - 2*n**2 + (n-1)*n/2
            'diagonal' - 
            'scalar' - 
        n: number of innovations in the model
    
    Returns
    -------
        A, B, C: (n, n) array, parameter matrices
    """
    A = theta[:n**2].reshape([n, n])
    B = theta[n**2:2*n**2].reshape([n, n])
    C = np.zeros((n, n))
    C[np.tril_indices(n)] = theta[2*n**2:]
    return A, B, C

def convert_abc_to_theta(A, B, C):
    """Convert parameter matrices A, B, and C to 1-dimensional array.
    
    Parameters
    ----------
        A, B, C: (n, n) arrays
            parameter matrices
    
    Returns
    -------
        1-dimensional array of parameters
            Length depends on the model restrictions:
            'full' - 2*n**2 + (n-1)*n/2
            'diagonal' - 
            'scalar' - 
    """
    theta = [A.flatten(), B.flatten(), C[np.tril_indices(C.shape[0])]]
    return np.concatenate(theta)

def convert_theta_to_ab(theta, n, restriction):
    """Convert 1-dimensional array of parameters to matrices A, and B.
    
    Parameters
    ----------
        theta: 1-dim array
            Parameters of the model
            Length depends on the model restrictions:
            'full' - 2*n**2 + (n-1)*n/2
            'diagonal' - 2*n
            'scalar' - 2
        n: int
            number of innovations in the model
        restriction: str
            can be 'full', 'diagonal', 'scalar'
    
    Returns
    -------
        A, B: (n, n) arrays, parameter matrices
    """
    if restriction == 'full':
        A = theta[:n**2].reshape([n, n])
        B = theta[n**2:].reshape([n, n])
    elif restriction == 'diagonal':
        A = np.diag(theta[:n])
        B = np.diag(theta[n:])
    elif restriction == 'scalar':
        A = np.eye(n) * theta[0]
        B = np.eye(n) * theta[1]
    else:
        # !!! Should raise exception "Wrong restriction'
        pass
    return A, B

def convert_ab_to_theta(A, B, restriction):
    """Convert parameter matrices A and B to 1-dimensional array.
    
    Parameters
    ----------
        A: (n, n) array
            Parameter matrix
        B: (n, n) array
            Parameter matrix
        restriction: str
            Can be 'full', 'diagonal', 'scalar'
    
    Returns
    -------
        1-dimensional array
            Parameters of the model
            Length depends on the model restrictions:
            'full' - 2*n**2
            'diagonal' - 2*n
            'scalar' - 2
    """
    if restriction == 'full':
        theta = [A.flatten(), B.flatten()]
    elif restriction == 'diagonal':
        theta = [np.diag(A), np.diag(B)]
    elif restriction == 'scalar':
        return np.array([A[0, 0], B[0, 0]])
    else:
        # !!! Should raise exception "Wrong restriction'
        pass
    return np.concatenate(theta)

def stationary_H(A, B, C):
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
    
def plot_data(u, H):
    """Plot time series of H and u elements.
    
    Parameters
    ----------
        u: (T, n) array
            innovations
        H: (T, n, n) array
            variance/covariances
    """
    T, n = u.shape
    fig, axes = plt.subplots(nrows = n**2, ncols = 1)
    for ax, i in zip(axes , range(n**2)):
        ax.plot(range(T), H.reshape([T, n**2])[:, i])
    plt.plot()
    
    fig, axes = plt.subplots(nrows = n, ncols = 1)
    for ax, i in zip(axes , range(n)):
        ax.plot(range(T), u[:, i])
    plt.plot()

if __name__ == '__main__':
    pass