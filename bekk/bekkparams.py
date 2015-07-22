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

    """Class to hold parameters of the BEKK model in different representations.

    Attributes
    ----------
    amat, bmat, cmat : (nstocks, nstocks) arrays
        Matrix representations of BEKK parameters
    theta : 1-dimensional array
        Vector of model parameters
    innov : (nobs, ntocks) array
            Return innovations
    restriction : str
        Can be
            - 'full'
            - 'diagonal'
            - 'scalar'
    target : (nstocks, nstocks) array
        Variance targeting flag. If True, then cmat is not returned.

    Methods
    -------
    unconditional_var
        Return unconditional variance given parameters
    constraint
        Constraint on parameters for stationarity
    log_string
        Forms the nice string for output

    """

    def __init__(self, nstocks=2):
        """Class constructor.

        Parameters
        ----------
        restriction : str
            Can be
                - 'full'
                - 'diagonal'
                - 'scalar'
        target : (nstocks, nstocks) arrays
            Variance targeting flag. If True, then cmat is not returned.
        innov : (nobs, ntocks) array
            Return innovations
        kwargs : keyword arguments, optional

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
        return self.__str__()

    @classmethod
    def from_abc(cls, amat=None, bmat=None, cmat=None):
        """Initialize from A, B, and C arrays.

        """
        nstocks = amat.shape[0]
        param = cls(nstocks)
        param.amat = amat
        param.bmat = bmat
        param.cmat = cmat
        return param

    @classmethod
    def from_target(cls, amat=None, bmat=None, target=None):
        """Initialize A, B, C from target.

        Notes
        -----
        amat, bmat, cmat : (nstocks, nstocks) arrays
            Parameter matrices
        theta : 1d array of parameters
            Length depends on the model restrictions and variance targeting
            If targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not targeting:
                - + (n-1)*n/2 for parameter C

        """
        nstocks = target.shape[0]
        if (amat is None) and (bmat is None):
            param = cls(nstocks)
            amat, bmat = param.amat, param.bmat
        cmat = cls.find_cmat(amat=amat, bmat=bmat, target=target)
        return cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)

    @staticmethod
    def find_cmat(amat=None, bmat=None, target=None):
        """Find C matrix given A, B, and H.
        Solve for C in H = CC' + AHA' + BHB' given A, B, H.

        """
        ccmat = target - amat.dot(target).dot(amat.T) \
            - bmat.dot(target).dot(bmat.T)

        # Extract C parameter
        try:
            return sl.cholesky(ccmat, 1)
        except sl.LinAlgError:
            warnings.warn('Matrix C is singular!')
            return None

    @classmethod
    def from_theta(cls, theta=None, nstocks=None,
                   restriction=None, target=None):
        """Initialize A, B, C from theta.

        Notes
        -----
        amat, bmat, cmat : (nstocks, nstocks) arrays
            Parameter matrices
        theta : 1d array of parameters
            Length depends on the model restrictions and variance targeting
            If targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not targeting:
                - + (n-1)*n/2 for parameter C

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

    def get_theta(self, restriction=None, var_target=False):
        """Convert parameter mratrices to 1-dimensional array.

        Notes
        -----
        amat, bmat, cmat : (nstocks, nstocks) arrays
            Parameter matrices
        theta : 1-dimensional array of parameters
            Length depends on the model restrictions and variance targeting
            If targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not targeting:
                - + (n-1)*n/2 for parameter cmat
r
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

    @staticmethod
    def find_stationary_var(amat=None, bmat=None, cmat=None):
        """Find fixed point of H = CC' + AHA' + BHB' given A, B, C.

        Returns
        -------
        hvarnew : (nstocks, nstocks) array
            Stationary variance matrix

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
        hvar : (nstocks, nstocks) array
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
        string : list
            List of strings

        """
        string = []
        string.append('Varinace targeting = ' + str(self.target))
        string.append('Model restriction = ' + str(self.restriction))
        string.append('Max eigenvalue = %.4f' % self.constraint())
        string.append('\nA =\n' + np.array_str(self.amat))
        string.append('\nB =\n' + np.array_str(self.bmat))
        string.append('\nC =\n' + np.array_str(self.cmat))
        string.append('\nH0 estim =\n'
                      + np.array_str(self.unconditional_var()))
        return string


if __name__ == '__main__':

    pass
