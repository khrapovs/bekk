import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.optimize import minimize
import cProfile
import numba as nb
import time

np.set_printoptions(precision = 2, suppress = True)

# BEKK model
# u(t)|H(t) ~ N(0,H(t))
# u(t) = e(t)H(t)^(1/2), e(t) ~ N(0,I)
# H(t) = E_{t-1}[u(t)u(t)']
# One lag, no asymmetries
# H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'

class BEKK(object):
    def __init__(self, u):
        self.u = u
        self.T, self.n = u.shape
        self.H0 = estimate_H0(u)

    def likelihood(self, theta):
        A, B = convert_theta_to_ab(theta, self.n)
        H = np.empty((self.T, self.n, self.n))
        
        H[0] = self.H0
        
        for t in range(1, self.T):
            H[t] = H[0]
            uu = self.u[t-1, np.newaxis].T * self.u[t-1]
            H[t] += A.dot(uu - H[0]).dot(A.T)
            H[t] += B.dot(H[t-1] - H[0]).dot(B.T)
        
        sumf = 0
        for t in range(self.T):
            f, bad = contribution(self.u[t], H[t])
            sumf += f
            if bad:
                break
        
        if np.isinf(sumf):
            return 1e10
        else:
            return sumf
    
    def callback(self, xk):
        self.it += 1
        A, B = convert_theta_to_ab(xk, self.n)
        
        start_like = self.likelihood(self.theta_start)
        current_like = self.likelihood(xk)
        true_like = self.likelihood(self.theta0)
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
    
    def optimize_like(self, theta0, nit):
        #ones = np.ones(len(theta0))
        #bounds = list(zip(-5*ones, 5*ones))
        # So far works:
        # Nelder-Mead, BFGS, L-BFGS-B, TNC
        # Works, but not so good:
        # CG, Powell
        self.theta_start = theta0
        self.xk_old = theta0
        self.it = 0
        self.time_start = time.time()
        self.time_old = time.time()
        res = minimize(self.likelihood, theta0,
                       method = 'L-BFGS-B',
                       callback = self.callback,
                       options = {'disp': False, 'maxiter' : int(nit)})
        return res

def simulate_BEKK(theta0, n = 2, T = 1000, log = 'bekk_log.txt'):
    """Simulate data.
    
    Returns:
        u: multivariate innovation matrix, T x n
    """
#    self.n = n
#    self.T = T
#    self.theta0 = theta0
    
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

def contribution(u, H):
    """Contribution to the log-likelihood function for each observation."""
    Heig = np.linalg.eigvals(H)
    Hdet = np.linalg.det(H)
    bad = np.any(np.isinf(H)) or Hdet>1e20 or Hdet<1e-5 or np.any(Heig<0)
    if bad:
        f = 1e10
    else:
        f = np.log(Hdet) + u.dot(np.linalg.inv(H)).dot(np.atleast_2d(u).T)
        f = float(f/2)
    return f, bad
    
def estimate_H0(u):
    T = u.shape[0]
    return u.T.dot(u) / T

def convert_theta_to_abc(theta, n):
    A = theta[:n**2].reshape([n, n])
    B = theta[n**2:2*n**2].reshape([n, n])
    C = np.zeros((n, n))
    C[np.tril_indices(n)] = theta[2*n**2:]
    return A, B, C

def convert_theta_to_ab(theta, n):
    A = theta[:n**2].reshape([n, n])
    B = theta[n**2:2*n**2].reshape([n, n])
    return A, B

def convert_abc_to_theta(A, B, C):
    theta = [A.flatten(), B.flatten(), C[np.tril_indices(C.shape[0])]]
    return np.concatenate(theta)

def stationary_H(A, B, C):
    i, norm = 0, 1e3
    Hold = np.eye(A.shape[0])
    while (norm > 1e-3) or (i < 1000):
        Hnew = C.dot(C.T) + A.dot(Hold).dot(A.T) + B.dot(Hold).dot(B.T)
        norm = np.linalg.norm(Hnew - Hold)
        Hold = Hnew[:]
        i += 1
    return Hnew
    
def plot_data(u, H):
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
    bekk.theta0 = theta[:2*n**2]
    bekk.log_file = log_file
    
    # Shift initial theta    
    #theta_AB = theta[:2*n**2]
    #theta0_AB = theta_AB - .1
    
    # Randomize initial theta
    theta0_AB = np.random.rand(2*n**2)/10
    print(bekk.likelihood(theta0_AB))
    
    # maximum number of iterations
    nit = 1e6
    # Start timer for the whole optimization
    time_old = time.time()
    # Estimate parameters
    result = bekk.optimize_like(theta0_AB, nit)
    # How much time did it take?
    time_delta = (time.time() - time_old) / 60
    # Convert parameter vector to matrices
    A, B = convert_theta_to_ab(result.x, n)
    # Print results
    with open(log_file, 'a') as texfile:
        texfile.write('\n' + str(result) + 2*'\n')
        texfile.write('Total time (minutes) = %.2f' % time_delta)

def test_real():
    pass
    
if __name__ == '__main__':
    test_simulate(n = 2, T = 100)
#    cProfile.run('test(n = 2, T = 100)')