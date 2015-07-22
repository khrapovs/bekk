#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEKK(1,1) parameter class
=========================

"""
from __future__ import print_function, division

import numpy as np
import scipy.linalg as sl
import scipy.optimize as sco

from .utils import estimate_h0, _bekk_recursion

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

    def __init__(self, restriction=None, target=None, nstocks=None,
                 amat=None, bmat=None, cmat=None, theta=None):
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
        # Defaults:
        self.amat, self.bmat, self.cmat = amat, bmat, cmat
        self.theta = theta
        self.restriction = restriction
        self.target = target

        if (cmat is not None) and (target is not None):
            msg = 'C matrix and variance target can not be used together!'
            raise ValueError(msg)
        elif target is not None:
            self.cmat = self.__find_cmat()

        nomat = (amat is not None) or (bmat is not None) or (cmat is not None)
        if (theta is not None) and nomat:
            msg = 'No need to specify A and/or B and/or C if theta is given!'
            raise ValueError(msg)

        if nstocks is not None:
            self.nstocks = nstocks
        elif amat is not None:
            self.nstocks = amat.shape[0]
        elif bmat is not None:
            self.nstocks = bmat.shape[0]
        elif cmat is not None:
            self.nstocks = cmat.shape[0]
        elif target is not None:
            self.nstocks = target.shape[0]

        if self.theta is not None:
            self.__convert_theta_to_abc()
        elif (self.amat is not None) and (self.bmat is not None):
            self.__convert_abc_to_theta()
        else:
            raise TypeError('Not enough arguments to initialize BEKKParams!')

    def __convert_theta_to_abc(self):
        """Convert 1-dimensional array of parameters to matrices.

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
        if self.restriction == 'full':
            chunk = self.nstocks**2
            sqsize = [self.nstocks, self.nstocks]
            self.amat = self.theta[:chunk].reshape(sqsize)
            self.bmat = self.theta[chunk:2*chunk].reshape(sqsize)
        elif self.restriction == 'diagonal':
            chunk = self.nstocks
            self.amat = np.diag(self.theta[:chunk])
            self.bmat = np.diag(self.theta[chunk:2*chunk])
        elif self.restriction == 'scalar':
            chunk = 1
            self.amat = np.eye(self.nstocks) * self.theta[:chunk]
            self.bmat = np.eye(self.nstocks) * self.theta[chunk:2*chunk]
        else:
            raise ValueError('This restriction is not supported!')

        if self.target:
            self.cmat = self.__find_cmat()
        else:
            self.cmat = np.zeros((self.nstocks, self.nstocks))
            self.cmat[np.tril_indices(self.nstocks)] = self.theta[2*chunk:]

    def __convert_abc_to_theta(self):
        """Convert parameter matrices to 1-dimensional array.

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

        """
        if self.restriction == 'full':
            self.theta = [self.amat.flatten(), self.bmat.flatten()]
        elif self.restriction == 'diagonal':
            self.theta = [np.diag(self.amat), np.diag(self.bmat)]
        elif self.restriction == 'scalar':
            self.theta = [[self.amat[0, 0]], [self.bmat[0, 0]]]
        else:
            raise ValueError('This restriction is not supported!')

        if self.target is None:
            self.theta.append(self.cmat[np.tril_indices(self.cmat.shape[0])])

        self.theta = np.concatenate(self.theta)

    def __find_cmat(self):
        """Solve for C in H = CC' + AHA' + BHB' given A, B, H.

        Parameters
        ----------
        stationary_var : (nstocks, nstocks) arrays
            Stationary variance matrix, H

        """
        stationary_var = self.unconditional_var()
        cmat_sq = 2*stationary_var - _bekk_recursion(self, stationary_var,
                                                      stationary_var,
                                                      stationary_var)
        # Extract C parameter
        try:
            return sl.cholesky(cmat_sq, 1)
        except sl.LinAlgError:
            return None

    def find_stationary_var(self):
        """Find fixed point of H = CC' + AHA' + BHB' given A, B, C.

        Returns
        -------
        hvarnew : (nstocks, nstocks) array
            Stationary variance matrix

        """
        hvarold = np.eye(self.amat.shape[0])
        ccmat = self.cmat.dot(self.cmat.T)
        fun = lambda x: _bekk_recursion(self, ccmat, x, x)
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                return sco.fixed_point(fun, hvarold)
        except RuntimeError:
            return None

    def unconditional_var(self):
        """Unconditional variance matrix regardless of the model.

        Returns
        -------
        hvar : (nstocks, nstocks) array
            Unconditional variance amtrix

        """
        if self.target is None:
            return self.find_stationary_var()
        else:
            return self.target

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
