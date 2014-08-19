import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from scipy.optimize import minimize
import numba as nb

# BEKK model
# u(t)|H(t) ~ N(0,H(t))
# u(t) = e(t)H(t)^(1/2), e(t) ~ N(0,I)
# H(t) = E_{t-1}[u(t)u(t)']
# One lag, no asymmetries
# H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'

def simulate_BEKK(theta):
    A, B, C = convert_theta_to_abc(theta)
    
    n, T = 2, 1000
    mean, cov = np.zeros(n), np.eye(n)
    
    constr = np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B))).max()
    print(constr)
    
    e = np.random.multivariate_normal(mean, cov, T)
    H = np.empty((T, n, n))
    H[0] = np.eye(2)
    u = np.zeros((T, n))

    for t in range(1, T):
        H[t] = C.dot(C.T) + A.dot(u[t-1, np.newaxis].T * u[t-1]).dot(A.T) \
            + B.dot(H[t-1]).dot(B.T)
        u[t] = np.sum(e[t] * sp.linalg.sqrtm(H[t]), 1)
    
    return u, H
    
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

def convert_theta_to_abc(theta):
    A = theta[:4].reshape([2,2])
    B = theta[4:8].reshape([2,2])
    C = np.array([[theta[8], 0], [theta[9], theta[10]]])
    return A, B, C

def convert_abc_to_theta(A, B, C):
    return np.concatenate([A.flatten(), B.flatten(), C.flatten()])

@nb.autojit
def likelihood(theta, u):
    T, n = u.shape
    A, B, C = convert_theta_to_abc(theta)
    H = np.empty((T, n, n))
    H0 = np.eye(2)
    
    for i in range(1, 100):
        H0 = C.dot(C.T) + A.dot(H0).dot(A.T) + B.dot(H0).dot(B.T)
    H[0] = H0[:]

    f = .5 * np.log(np.linalg.det(H[0])) \
        + .5 * u[0].dot(np.linalg.inv(H[0])).dot(u[0, np.newaxis].T)
    
    for t in range(1, T):
        H[t] = C.dot(C.T) + A.dot(u[t-1, np.newaxis].T * u[t-1]).dot(A.T) \
            + B.dot(H[t-1]).dot(B.T)
        f += .5 * np.log(np.linalg.det(H[t])) \
            + .5 * u[t].dot(np.linalg.inv(H[t])).dot(u[t, np.newaxis].T)

    return float(f)

def optimize_like(u, theta0):
    
    res = minimize(likelihood, theta0, args = (u,),
                   method = 'Nelder-Mead',
                   options = {'disp': True, 'maxiter' : 500})
    return res
    
if __name__ == '__main__':
    np.set_printoptions(precision = 3, suppress = True)
    
    # A, B, C - n x n matrices
    A = np.array([[.15, 0], [0, .15]])
    B = np.array([[.8, 0], [0, .8]])
    C = np.array([1, .5, 1])
    theta = convert_abc_to_theta(A, B, C)
    
    u, H = simulate_BEKK(theta)
    
    print('Likelihood for true theta = %.2f' % likelihood(theta, u))
    
    result = optimize_like(u, theta*1.1)
    print(result)
    A, B, C = convert_theta_to_abc(result.x)
    print(A, 2*'\n', B, 2*'\n', C)