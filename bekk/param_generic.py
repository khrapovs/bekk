#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic parameter class
-----------------------

"""
from __future__ import print_function, division

import warnings

import numpy as np
import scipy.linalg as sl
import scipy.optimize as sco

__all__ = ['ParamGeneric']


class ParamGeneric(object):

    """Class to hold parameters of the BEKK model.

    Attributes
    ----------
    amat, bmat, cmat
        Matrix representations of BEKK parameters

    Methods
    -------
    from_abc
        Initialize from A, B, and C arrays
    find_cmat
        Find C matrix given A, B, and H
    find_stationary_var
        Return unconditional variance given parameter matrices
    get_uvar
        Return unconditional variance
    constraint
        Constraint on parameters for stationarity

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
        show = "\nA = \n" + str(self.amat)
        show += "\nB = \n" + str(self.bmat)
        show += "\nC = \n" + str(self.cmat)
        uvar = self.get_uvar()
        if uvar is not None:
            show += '\n\nUnconditional variance =\n' + np.array_str(uvar)
        else:
            show += '\nCould not compute unconditional variance!'
        show += '\n\nMax eigenvalue = %.4f' % self.constraint()
        return show + '\n'

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
            warnings.warn('Could not find stationary varaince!')
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
