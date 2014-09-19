# Check this repo for related R library: https://github.com/vst/mgarch/
# Alternative optimization library: http://www.pyopt.org/

from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.optimize import minimize
import scipy.linalg as sl
#import cProfile
#import numba as nb
import time
#from IPython.parallel import Client

np.set_printoptions(precision = 4, suppress = True)

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
        # How much time did it take?
        self.time_final = time.time()
        self.print_results()
        
def simulate_BEKK(theta0, n = 2, T = 1000, log = 'bekk_log.txt'):
    """Simulate data.
    
    Parameters
    ----------
    
    Returns
    -------
        u: (T, n) array
            multivariate innovation matrix
    """
    
    A, B, C = convert_theta_to_abc(theta0, n)
    mean, cov = np.zeros(n), np.eye(n)
    
    constr = np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B))).max()
    with open(log, 'a') as texfile:
        texfile.write('Max eigenvalue = %.2f' % constr)
    
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
    # Old version
#    Heig = sl.eigvals(H)
#    Hdet = sl.det(H)
#    bad = np.any(np.isinf(H)) or Hdet>1e20 or Hdet<1e-20 or Heig.any()<0
#    if bad:
#        f = 1e10
#    else:
#        f = np.log(Hdet) + u.dot(sl.inv(H)).dot(np.atleast_2d(u).T)
#        f = float(f/2)

    try:
        LU, piv = sl.lu_factor(H)
    except (sl.LinAlgError, ValueError):
        return 1e10, True
    
    Hdet = np.abs(np.prod(np.diag(LU)))
    if Hdet>1e20 or Hdet<1e-20:
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
        u: (T, n) array, inovations
    
    Returns
    -------
        E[u'u], (n, n) array
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
        A, B, C: (n, n) arrays, parameter matrices
    
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
        theta: array of parameters
            Length depends on the model restrictions:
            'full' - 2*n**2 + (n-1)*n/2
            'diagonal' - 2*n
            'scalar' - 2
        n: number of innovations in the model
        restriction: can be 'full', 'diagonal', 'scalar'
    
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
        A, B: (n, n) arrays, parameter matrices
        restriction: can be 'full', 'diagonal', 'scalar'
    
    Returns
    -------
        1-dimensional array of parameters
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
        A, B, C: (n, n) arrays, parameter matrices
    
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
        u: (T, n) array, innovations
        H: (T, n, n) array, variance/covariances
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

def test_simulate(n = 2, T = 100):
    log_file = 'bekk_log.txt'
    with open(log_file, 'w') as texfile:
        texfile.write('')
        
    # A, B, C - n x n matrices
    A = np.eye(n) * .25
    B = np.eye(n) * .95
    C = sp.linalg.cholesky(np.ones((n,n)) * .5 + np.eye(n) * .5, 1)
    theta = convert_abc_to_theta(A, B, C)
    
    # Simulate data    
    u = simulate_BEKK(theta, n = n, T = T, log = log_file)
    
    # Initialize the object
    bekk = BEKK(u)
    bekk.theta_true = theta[:2*n**2]
    bekk.log_file = log_file
    
    # Shift initial theta    
    #theta_AB = theta[:2*n**2]
    #theta0_AB = theta_AB - .1
    
    # Randomize initial theta
    theta0_AB = np.random.rand(2*n**2)/10
    
    # maximum number of iterations
    nit = 1e6
    # Start timer for the whole optimization
    time_old = time.time()
    # Estimate parameters
    result = bekk.estimate(theta0_AB, nit)
    # How much time did it take?
    time_delta = (time.time() - time_old) / 60
    # Convert parameter vector to matrices
    A, B = convert_theta_to_ab(result.x, n)
    # Print results
    with open(log_file, 'a') as texfile:
        texfile.write('\n' + str(result) + 2*'\n')
        texfile.write('Total time (minutes) = %.2f' % time_delta)

def regenerate_data(u_file):
    """Download and save data to disk.
    
    Parameters
    ----------
        u_file: str, name of the file to save to
    """
    import Quandl
    import pandas as pd
    
    token = 'ECK8bso5CLKnNui4kNpk'
    tickers = ["GOOG/NASDAQ_MSFT", "GOOG/NASDAQ_AAPL"]
    tickers2 = ["GOOG/NYSE_XOM", "GOOG/NYSE_OXY"]
    tickers3 = ["GOOG/NYSE_TGT", "GOOG/NYSE_WMT"]
    tickers.extend(tickers2)
    tickers.extend(tickers3)
    prices = []
    for tic in tickers:
        df = Quandl.get(tic, authtoken = token,
                        trim_start = "2000-01-01",
                        trim_end = "2007-12-31")[['Close']]
        df.rename(columns = {'Close' : tic}, inplace = True)
        prices.append(df)
    prices = pd.concat(prices, axis = 1)
    
    ret = (np.log(prices) - np.log(prices.shift(1))) * 100
    ret.dropna(inplace = True)
    ret.plot()
    plt.show()
    
    # Create array of innovations    
    u = np.array(ret.apply(lambda x: x - x.mean()))#[-2000:]
    print(u.shape)
    np.save(u_file, u)
    np.savetxt(u_file[:-4] + '.csv', u, delimiter = ",")

def init_parameters(restriction, n):
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
    # Randomize initial theta
    #theta0 = np.random.rand(2*n**2)/10
    # Clever initial theta
    # A, B - n x n matrices
    A = np.eye(n) * .15 # + np.ones((n, n)) *.05
    B = np.eye(n) * .95
    theta = convert_ab_to_theta(A, B, restriction)
    return theta
    
def test_real(method, theta_start, restriction, stage):
    # Load data    
    u_file = 'innovations.npy'
    # Load data from the drive
    u = np.load(u_file)
    
    #import numpy as np
    log_file = restriction + '/bekk_log_' + method + '_' + str(stage) + '.txt'
    # Clean log file
    with open(log_file, 'w') as texfile:
        texfile.write('')
    # Initialize the object
    bekk = BEKK(u)
    # Restriction of the model, 'scalar', 'diagonal', 'full'
    bekk.restriction = restriction
    # maximum number of iterations
    bekk.maxiter = 1e6
    # Print results for each iteration?
    bekk.use_callback = False
    # Set log file name
    bekk.log_file = log_file
    # Do we want to download data and overwrite it on the drive?
    #regenerate_data(u_file)
    
    # 'Newton-CG', 'dogleg', 'trust-ncg' require gradient
    #    methods = ['Nelder-Mead','Powell','CG','BFGS','L-BFGS-B',
    #               'TNC','COBYLA','SLSQP']
    # Optimization method
    bekk.method = method
    # Estimate parameters
    bekk.estimate(theta_start)
    # bekk.estimate(bekk.theta_final)
    return bekk

def two_stage_estimation():
    """Estiamte BEKK using different optimization methods in two steps.
    
    Choose the best method according to the reached likelihood.'
    Take the final theta from this method.'
    Use it as a starting point for the next stage.
    
    """
    # Load data    
    u_file = 'innovations.npy'
    # Load data from the drive
    u = np.load(u_file)
    n = u.shape[1]
    # Choose the model restriction: scalar, diagonal, full
    restriction = 'scalar'
    # Initialize parameters
    theta_start = init_parameters(restriction, n)
    print(theta_start)
    from multiprocessing import Pool
#    'Newton-CG', 'dogleg', 'trust-ncg' require gradient
    # 'BFGS', 'CG', 'Nelder-Mead' do a very poor job.
    methods = ['Powell','L-BFGS-B','TNC','COBYLA','SLSQP']
    #methods = ['L-BFGS-B']
    with Pool(processes=len(methods)) as pool:
        results = pool.starmap(test_real,
                               zip(methods,
                                   [theta_start for x in range(len(methods))],
                                   [restriction for x in range(len(methods))],
                                   [1 for x in range(len(methods))]))
        pool.close()
    
    likes = [res.res.fun for res in results]
    thetas = [res.res.x for res in results]
    theta_start = thetas[likes.index(min(likes))]
    methods = ['Powell','L-BFGS-B','TNC','COBYLA','SLSQP']
    print(theta_start)
    with Pool(processes=len(methods)) as pool:
        results = pool.starmap(test_real,
                               zip(methods,
                                   [theta_start for x in range(len(methods))],
                                   [restriction for x in range(len(methods))],
                                   [2 for x in range(len(methods))]))
        pool.close()

def one_stage_estimation():
    """Estiamte BEKK using different optikization methods in one step.
    """
    # Load data    
    u_file = 'innovations.npy'
    # Load data from the drive
    u = np.load(u_file)
    n = u.shape[1]
    # Choose the model restriction: scalar, diagonal, full
    restriction = 'diagonal'
    # Initialize parameters
    theta_start = init_parameters(restriction, n)
    print(theta_start)
    # 'Newton-CG', 'dogleg', 'trust-ncg' require gradient
    # 'BFGS', 'CG', 'Nelder-Mead' do a very poor job.
    methods = ['Powell','L-BFGS-B','TNC','COBYLA','SLSQP']
    for method in methods:
        test_real(method, theta_start, restriction, 1)
    
if __name__ == '__main__':
#    test_simulate(n = 2, T = 100)
#    cProfile.run('test(n = 2, T = 100)')
#    bekk = test_real('TNC')
    
    one_stage_estimation()