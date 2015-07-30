#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEKK(1,1) parameter class
=========================

"""
from __future__ import print_function, division

import warnings

import numpy as np
import scipy.linalg as sl
import scipy.optimize as sco

__all__ = ['BEKKParams']


class BEKKParams(object):

    """Class to hold parameters of the BEKK model.

    Attributes
    ----------
    amat, bmat, cmat : (nstocks, nstocks) arrays
        Matrix representations of BEKK parameters

    Methods
    -------
    from_abc
        Initialize from A, B, and C arrays
    from_target
        Initialize from A, B, and variance target
    from_theta
        Initialize from theta vector
    get_theta
        Convert parameter mratrices to 1-dimensional array
    find_cmat
        Find C matrix given A, B, and H
    find_stationary_var
        Return unconditional variance given parameter matrices
    get_uvar
        Return unconditional variance
    constraint
        Constraint on parameters for stationarity
    log_string
        Forms the nice string for output

    """

    def __init__(self, nstocks=2):
        """Class constructor.

        Parameters
        ----------
        nstocks : int
            Number os stocks in the model

        """
        self.amat = np.eye(nstocks) * .1**.5
        self.bmat = np.eye(nstocks) * .8**.5
        self.cmat = self.find_cmat(amat=self.amat, bmat=self.bmat,
                                   target=np.eye(nstocks))

    def __str__(self):
        """String representation.

        """
        show = "A = \n" + str(self.amat)
        show += "\nB = \n" + str(self.bmat)
        show += "\nC = \n" + str(self.cmat)
        return show

    def __repr__(self):
        """String representation.

        """
        return self.__str__()

    @classmethod
    def from_abc(cls, amat=None, bmat=None, cmat=None):
        """Initialize from A, B, and C arrays.

        Parameters
        ----------
        amat, bmat, cmat : (nstocks, nstocks) arrays
            Parameter matrices

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        nstocks = amat.shape[0]
        param = cls(nstocks)
        param.amat = amat
        param.bmat = bmat
        param.cmat = cmat
        return param

    @classmethod
    def from_target(cls, amat=None, bmat=None, target=None):
        """Initialize from A, B, and variance target.

        Parameters
        ----------
        amat, bmat, target : (nstocks, nstocks) arrays
            Parameter matrices

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        nstocks = target.shape[0]
        if (amat is None) and (bmat is None):
            param = cls(nstocks)
            amat, bmat = param.amat, param.bmat
        cmat = cls.find_cmat(amat=amat, bmat=bmat, target=target)
        return cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)

    @classmethod
    def from_spatial(cls, avecs=None, bvecs=None, dvecs=None,
                     vvec=None, weights=None):
        """Initialize from spatial representation.

        Parameters
        ----------
        avecs, bvecs: (ncat+1, nstocks) arrays
            Parameter matrices
        dvecs : (ncat, nstocks) array
            Parameter matrices
        vvec : (nstocks, ) array
            Parameter vector
        weights : (ncat, nstocks, nstocks) array
            Weight matrices

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        ncat, nstocks = weights.shape[:2]
        amat, bmat, dmat = cls.find_abdmat_spatial(avecs=avecs, bvecs=bvecs,
                                                   dvecs=dvecs,
                                                   weights=weights)
        dmat_inv = sl.inv(dmat)
        cmat = dmat_inv.dot(np.diag(vvec)).dot(dmat_inv)
        param = cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        param.avecs = avecs
        param.bvecs = bvecs
        param.dvecs = dvecs
        param.vvec = vvec
        param.weights = weights
        return param

    @staticmethod
    def find_abdmat_spatial(avecs=None, bvecs=None, dvecs=None, weights=None):
        """Initialize amat, bmat, and dmat from spatial representation.

        Parameters
        ----------
        avecs, bvecs: (ncat+1, nstocks) arrays
            Parameter matrices
        dvecs : (ncat, nstocks) array
            Parameter matrices
        weights : (ncat, nstocks, nstocks) array
            Weight matrices

        Returns
        -------
        amat, bmat, dmat : (nstocks, nstocks) arrays
            BEKK parameter matrices

        """
        ncat, nstocks = weights.shape[:2]
        amat, bmat = np.diag(avecs[0]), np.diag(bvecs[0])
        dmat = np.eye(nstocks)
        for i in range(ncat):
            amat += np.diag(avecs[i+1]).dot(weights[i])
            bmat += np.diag(bvecs[i+1]).dot(weights[i])
            dmat -= np.diag(dvecs[i]).dot(weights[i])
        return amat, bmat, dmat

    @staticmethod
    def find_cmat(amat=None, bmat=None, target=None):
        """Find C matrix given A, B, and H.
        Solve for C in H = CC' + AHA' + BHB' given A, B, H.

        Parameters
        ----------
        amat, bmat, target : (nstocks, nstocks) arrays
            Parameter matrices

        Returns
        -------
        (nstocks, nstocks) array
            Cholesky decomposition of CC'

        """
        ccmat = target - amat.dot(target).dot(amat.T) \
            - bmat.dot(target).dot(bmat.T)

        # Extract C parameter
        try:
            return sl.cholesky(ccmat, 1)
        except sl.LinAlgError:
            warnings.warn('Matrix C is singular!')
            return None

    @staticmethod
    def find_vvec(avecs=None, bvecs=None, dvecs=None,
                  weights=None, target=None):
        r"""Find v vector given a, b, d, H, and weights.
        Solve for diagonal of

        .. math::
            D_{W}\left(H-A_{W}^{0}HA_{W}^{0\prime}
            -B_{W}^{0}HB_{W}^{0\prime}\right)D_{W}^{\prime}

        Parameters
        ----------
        avecs, bvecs: (ncat+1, nstocks) arrays
            Parameter matrices
        dvecs : (ncat, nstocks) array
            Parameter matrices
        vvec : (nstocks, ) array
            Parameter vector
        weights : (ncat, nstocks, nstocks) array
            Weight matrices
        target : (nstocks, nstocks) array
            Unconditional variance matrix

        Returns
        -------
        (nstocks, ) array
            Vector v

        """
        mats = BEKKParams.find_abdmat_spatial(avecs=avecs, bvecs=bvecs,
                                              dvecs=dvecs, weights=weights)
        amat, bmat, dmat = mats
        ccmat = target - amat.dot(target).dot(amat.T) \
            - bmat.dot(target).dot(bmat.T)
        return np.diag(dmat.dot(ccmat).dot(dmat.T))

    @classmethod
    def from_theta(cls, theta=None, nstocks=None,
                   restriction='scalar', target=None):
        """Initialize from theta vector.

        Parameters
        ----------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If target is not None:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2

            If target is None:
                - + (n-1)*n/2 for parameter C
        nstocks : int
            Number of stocks in the model
        restriction : str
            Can be
                - 'full'
                - 'diagonal'
                - 'scalar'
        target : (nstocks, nstocks) array
            Variance target matrix

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        if restriction == 'full':
            chunk = nstocks**2
            sqsize = [nstocks, nstocks]
            amat = theta[:chunk].reshape(sqsize)
            bmat = theta[chunk:2*chunk].reshape(sqsize)
        elif restriction == 'diagonal':
            chunk = nstocks
            amat = np.diag(theta[:chunk])
            bmat = np.diag(theta[chunk:2*chunk])
        elif restriction == 'scalar':
            chunk = 1
            amat = np.eye(nstocks) * theta[:chunk]
            bmat = np.eye(nstocks) * theta[chunk:2*chunk]
        else:
            raise ValueError('This restriction is not supported!')

        if target is not None:
            cmat = cls.find_cmat(amat=amat, bmat=bmat, target=target)
        else:
            cmat = np.zeros((nstocks, nstocks))
            cmat[np.tril_indices(nstocks)] = theta[2*chunk:]

        return cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)

    @classmethod
    def from_theta_spatial(cls, theta=None, weights=None, target=None):
        """Initialize from theta vector.

        Parameters
        ----------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If var_target:
                - 3*n*(m+1) - n

            If not var_target:
                - + n
        weights : (ncat, nstocks, nstocks) array
            Weight matrices
        target : (nstocks, nstocks) array
            Variance target matrix

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        ncat, nstocks = weights.shape[:2]
        abvecs_size = (ncat+1)*nstocks
        dvecs_size = ncat*nstocks
        avecs = theta[:abvecs_size].reshape((ncat+1, nstocks))
        bvecs = theta[abvecs_size:2*abvecs_size].reshape((ncat+1, nstocks))
        dvecs = theta[2*abvecs_size:2*abvecs_size+dvecs_size]
        dvecs = dvecs.reshape((ncat, nstocks))

        if target is not None:
            if theta.size != 3*nstocks*(ncat+1)-nstocks:
                raise ValueError('Incorrect number of params for targeting!')
            vvec = cls.find_vvec(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                 weights=weights, target=target)
        else:
            if theta.size != 3*nstocks*(ncat+1):
                msg = 'Incorrect number of params for no targeting!'
                raise ValueError(msg)
            vvec = theta[2*abvecs_size+dvecs_size:]

        return cls.from_spatial(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                vvec=vvec, weights=weights)

    def get_theta(self, restriction='scalar', var_target=True):
        """Convert parameter mratrices to 1-dimensional array.

        Parameters
        ----------
        restriction : str
            Can be
                - 'full'
                - 'diagonal'
                - 'scalar'
        var_target : bool
            Whether to estimate only A and B (True) or C as well (False)

        Returns
        -------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If var_target:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2

            If not var_target:
                - + (n-1)*n/2 for parameter cmat

        """
        if restriction == 'full':
            theta = [self.amat.flatten(), self.bmat.flatten()]
        elif restriction == 'diagonal':
            theta = [np.diag(self.amat), np.diag(self.bmat)]
        elif restriction == 'scalar':
            theta = [[self.amat[0, 0]], [self.bmat[0, 0]]]
        else:
            raise ValueError('This restriction is not supported!')

        if not var_target:
            theta.append(self.cmat[np.tril_indices(self.cmat.shape[0])])

        return np.concatenate(theta)

    def get_theta_spatial(self, var_target=True):
        """Convert parameter mratrices to 1-dimensional array.

        Parameters
        ----------
        var_target : bool
            Whether to estimate only a, b, and d (True) or v as well (False)

        Returns
        -------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If var_target:
                - 3*n*(m+1) - n

            If not var_target:
                - + n

        """
        theta = [self.avecs.flatten(), self.bvecs.flatten(),
                 self.dvecs.flatten()]

        if not var_target:
            theta.append(self.vvec)

        return np.concatenate(theta)

    @staticmethod
    def find_stationary_var(amat=None, bmat=None, cmat=None):
        """Find fixed point of H = CC' + AHA' + BHB' given A, B, C.

        Parameters
        ----------
        amat, bmat, cmat : (nstocks, nstocks) arrays
            Parameter matrices

        Returns
        -------
        (nstocks, nstocks) array
            Unconditional variance matrix

        """
        hvar = np.eye(amat.shape[0])
        ccmat = cmat.dot(cmat.T)
        fun = lambda x: 2 * x - ccmat - amat.dot(x).dot(amat.T) \
            - bmat.dot(x).dot(bmat.T)
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sco.fixed_point(fun, hvar)
        except RuntimeError:
            return None

    def get_uvar(self):
        """Unconditional variance matrix regardless of the model.

        Returns
        -------
        (nstocks, nstocks) array
            Unconditional variance amtrix

        """
        return self.find_stationary_var(amat=self.amat, bmat=self.bmat,
                                        cmat=self.cmat)

    def constraint(self):
        """Compute the largest eigenvalue of BEKK model.

        Returns
        -------
        float
            Largest eigenvalue

        """
        kron_a = np.kron(self.amat, self.amat)
        kron_b = np.kron(self.bmat, self.bmat)
        return np.abs(sl.eigvals(kron_a + kron_b)).max()

    def log_string(self):
        """Create string for log file.

        Returns
        -------
        string : str
            Formatted output text

        """
        string = self.__str__()
        string += '\nMax eigenvalue = %.4f' % self.constraint()
        string += '\nUnconditional variance=\n' + np.array_str(self.get_uvar())
        return string


if __name__ == '__main__':

    pass
