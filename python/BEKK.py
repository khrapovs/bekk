import numpy as np
import scipy as sp
import matplotlib.pylab as plt

# BEKK model
# u(t)|H(t) ~ N(0,H(t))
# u(t) = e(t)H(t)^(1/2), e(t) ~ N(0,I)
# H(t) = E_{t-1}[u(t)u(t)']
# One lag, no asymmetries
# H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'

def simulate_BEKK():
    n, T = 2, 100
    mean, cov = np.zeros(n), np.eye(n)
    # A, B, C - n x n matrices
    A = np.array([[.5, .5], [.2, .2]])
    B = np.array([[.5, .5], [.2, .2]])
    C = np.array([[.5, .5], [.2, .2]])
    
    constr = np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B))) - .999
    print(constr)
    
    e = np.random.multivariate_normal(mean, cov, T)
    H = np.empty((T, n, n))
    H[0] = np.eye(2)
    u = np.zeros((T, n))

    for t in range(1, T):
        print(t)
        H[t] = C.dot(C.T) + A.dot(u[t-1, np.newaxis].T * u[t-1]).dot(A.T) \
            + B.dot(H[t-1]).dot(B.T)
        u[t] = np.sum(e[t] * sp.linalg.sqrtm(H[t]), 1)
    
    fig, axes = plt.subplots(nrows = n**2, ncols = 1)
    for ax, i in zip(axes , range(n**2)):
        ax.plot(range(T), H.reshape([T, n**2])[:, i])
    plt.plot()
    
    return u, H

if __name__ == '__main__':
    u, H = simulate_BEKK()
    