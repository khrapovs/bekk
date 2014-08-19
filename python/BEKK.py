import numpy as np
import scipy as sp
import matplotlib.pylab as plt

# BEKK model
# u(t)|H(t) ~ N(0,H(t))
# u(t) = e(t)H(t)^(1/2), e(t) ~ N(0,I)
# H(t) = E_{t-1}[u(t)u(t)']
# One lag, no asymmetries
# H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'

def simulate_BEKK(A, B, C):
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
    
    fig, axes = plt.subplots(nrows = n**2, ncols = 1)
    for ax, i in zip(axes , range(n**2)):
        ax.plot(range(T), H.reshape([T, n**2])[:, i])
    plt.plot()
    
    fig, axes = plt.subplots(nrows = n, ncols = 1)
    for ax, i in zip(axes , range(n)):
        ax.plot(range(T), u[:, i])
    plt.plot()
    
    return u, H

def convert_param(theta):
    A = theta[:4].reshape([2,2])
    B = theta[4:8].reshape([2,2])
    C = theta[8:].reshape([2,2])
    return A, B, C

def likelihood(theta, u):
    T, n = u.shape
    A, B, C = convert_param(theta)
    f = 0
    H = np.empty((T, n, n))
    H0 = np.eye(2)
    for i in range(1, 100):
        H0 = C.dot(C.T) + A.dot(H0).dot(A.T) + B.dot(H0).dot(B.T)
    print(H0)
    H[0] = H0[:]
    for t in range(1, T):
        H[t] = C.dot(C.T) + A.dot(u[t-1, np.newaxis].T * u[t-1]).dot(A.T) \
            + B.dot(H[t-1]).dot(B.T)
        f += -.5 * np.log(np.linalg.det(H[t])) \
            - .5 * u[t].dot(np.linalg.inv(H[t])).dot(u[t, np.newaxis].T)
    return float(f)

if __name__ == '__main__':

    # A, B, C - n x n matrices
    A = np.array([[.15, 0], [0, .15]])
    B = np.array([[.8, 0], [0, .8]])
    C = np.array([[1, 0], [.5, 1]])

    u, H = simulate_BEKK(A, B, C)
    theta = np.concatenate([A.flatten(), B.flatten(), C.flatten()])
    
    print(likelihood(theta, u))