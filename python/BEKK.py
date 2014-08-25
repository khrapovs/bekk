import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.optimize import minimize
import cProfile
import numba as nb

# BEKK model
# u(t)|H(t) ~ N(0,H(t))
# u(t) = e(t)H(t)^(1/2), e(t) ~ N(0,I)
# H(t) = E_{t-1}[u(t)u(t)']
# One lag, no asymmetries
# H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'

class BEKK(object):
    def __init__(self):
        self.H = None
        self.u = None

    def simulate_BEKK(self, theta0, n = 2, T = 1000):
        self.n = n
        self.T = T
        self.theta0 = theta0
        
        A, B, C = convert_theta_to_abc(theta0, n)
        self.A0, self.B0, self.C0 = A, B, C
        
        mean, cov = np.zeros(n), np.eye(n)
        
        constr = np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B))).max()
        print('Max eigenvalue = ', constr)
        
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
        
        self.u = u
        self.H0 = estimate_H0(u)
    
    def likelihood(self, theta):
        A, B = convert_theta_to_ab(theta, self.n)
        H = np.empty((self.T, self.n, self.n))
        
        #H[0] = stationary_H(A, B, C)
        H[0] = self.H0
        
        for t in range(1, self.T):
            H[t] = H[0]
            uu = self.u[t-1, np.newaxis].T * self.u[t-1]
            H[t] += A.dot(uu - H[0]).dot(A.T)
            H[t] += B.dot(H[t-1] - H[0]).dot(B.T)
        
        f = 0
        for t in range(0, self.T):
            f += contribution(self.u[t], H[t])
        
        if np.isinf(f):
            return 1e10
        else:
            return f
    
    def callback(self, xk):
        n = (len(xk) / 2) ** .5
        A, B = convert_theta_to_ab(xk, n)
        print('A = \n', A, '\nB = \n', B, '\n')
        delta = self.likelihood(xk) - self.likelihood(self.theta0)
        print('Current likelihood = %.2f' % self.likelihood(xk))
        print('Current delta = %.2f' % delta)
    
    def optimize_like(self, theta0, nit):
    #    ones = np.ones(len(theta0))
    #    bounds = list(zip(-5*ones, 5*ones))
        # So far works:
        # Nelder-Mead, BFGS, L-BFGS-B
        # Works, but not so good:
        # CG
        self.theta_start = theta0
        res = minimize(self.likelihood, theta0,
                       method = 'Powell',
                       callback = self.callback,
                       options = {'disp': True, 'maxiter' : int(nit)})
        return res

def contribution(u, H):
    """Contribution to the log-likelihood function for each observation."""
    Hdet = np.linalg.det(H)
    bad = np.any(np.isinf(H)) or Hdet > 1e20 or Hdet < 1e-5
    if bad:
        return 1e10
    else:
        # To be absolutely correct, it must be multiplied by .5
        f = np.log(np.linalg.det(H))
        f += u.dot(np.linalg.inv(H)).dot(np.atleast_2d(u).T)
        return float(f)
    
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

def test(n = 2, T = 100):
    # A, B, C - n x n matrices
    A = np.eye(n) * .25
    B = np.eye(n) * .95
    C = sp.linalg.cholesky(np.ones((n,n)) * .5 + np.eye(n) * .5, 1)
    
    theta = convert_abc_to_theta(A, B, C)
    theta_AB = theta[:2*n**2]
    
    bekk = BEKK()
    bekk.simulate_BEKK(theta, n = n, T = T)
    
    print('Likelihood for true theta = %.2f' % bekk.likelihood(theta_AB))
    theta0_AB = theta_AB - .1
    print('Likelihood for initial theta = %.2f' % bekk.likelihood(theta0_AB))
    nit = 1e6
    result = bekk.optimize_like(theta0_AB, nit)
    print(result)
    A, B = convert_theta_to_ab(result.x, n)
    print(A, 2*'\n', B)


if __name__ == '__main__':
    np.set_printoptions(precision = 2, suppress = True)
    test(n = 6, T = 2000)
#    cProfile.run('test(2)')