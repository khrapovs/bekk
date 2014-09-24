from __future__ import print_function, division

import numpy as np
import scipy as sp
import time
import matplotlib.pylab as plt

from bekk import BEKK, simulate_BEKK
from bekk import convert_abc_to_theta, convert_theta_to_ab
from bekk import convert_ab_to_theta

np.set_printoptions(precision=4, suppress=True)

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
    
    token = open('Quandl.token', 'r').read()
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
    log_file = 'bekk_log_' + method + '_' + str(stage) + '.txt'
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

def simple_test():
    """Estimate BEKK.
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
    
    # Run first stage estimation
    test_real('L-BFGS-B', theta_start, restriction, 1)
 
if __name__ == '__main__':
    simple_test()