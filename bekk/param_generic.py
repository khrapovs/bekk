#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic parameter class
-----------------------

"""
from __future__ import print_function, division

import warnings

from functools import partial

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
    from_target
        Initialize from A, B, and variance target
    find_stationary_var
        Return unconditional variance given parameter matrices
    get_uvar
        Return unconditional variance
    constraint
        Constraint on parameters for stationarity

    """

    def __init__(self, nstocks=2, abstart=(.1, .85), target=None):
        """Class constructor.

        Parameters
        ----------
        nstocks : int
            Number os stocks in the model

        """
        self.amat = np.eye(nstocks) * abstart[0]**.5
        self.bmat = np.eye(nstocks) * abstart[1]**.5
        if target is None:
            target = np.eye(nstocks)
        self.cmat = self.find_cmat(amat=self.amat, bmat=self.bmat,
                                   target=target)

    def __str__(self):
        """String representation.

        """
        show = '\n\nMax eigenvalue = %.4f' % self.constraint()
        show += '\nPenalty = %.4f\n' % self.penalty()

        show += "\nA =\n" + str(self.amat)
        show += "\nB =\n" + str(self.bmat)
        show += "\nC =\n" + str(self.cmat)

        if self.get_model() == 'spatial':
            show += '\n\nSpatial parameters:'
            show += '\na =\n' + str(self.avecs)
            show += '\nb =\n' + str(self.bvecs)
            show += '\nd =\n' + str(self.dvecs)

        uvar = self.get_uvar()
        if uvar is not None:
            show += '\n\nUnconditional variance =\n' + np.array_str(uvar)
        else:
            show += '\n\nCould not compute unconditional variance!'
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
#            alpha = - np.min([0, sl.eigvals(ccmat).min().real * 1.1])
#            ridge = np.eye(ccmat.shape[0]) * alpha
            return sl.cholesky(ccmat, 1)
        except sl.LinAlgError:
            # warnings.warn('Matrix C is singular!')
            return np.zeros_like(ccmat)
            # return None

    @staticmethod
    def fixed_point(uvar, amat=None, bmat=None, ccmat=None):
        """Function for finding fixed point of
        H = CC' + AHA' + BHB' given A, B, C.

        Parameters
        ----------
        uvar : 1d array
            Lower triangle of symmetric variance matrix
        amat, bmat, ccmat : (nstocks, nstocks) arrays
            Parameter matrices

        Returns
        -------
        (nstocks, nstocks) array

        """
        nstocks = amat.shape[0]
        hvar = np.zeros((nstocks, nstocks))
        hvar[np.tril_indices(nstocks)] = uvar
        hvar[np.triu_indices(nstocks, 1)] = hvar[np.tril_indices(nstocks, -1)]
        diff = 2 * hvar - ccmat - amat.dot(hvar).dot(amat.T) \
            - bmat.dot(hvar).dot(bmat.T)
        return diff[np.tril_indices(nstocks)]

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
        nstocks = amat.shape[0]
        kwargs = {'amat': amat, 'bmat': bmat, 'ccmat': cmat.dot(cmat.T)}
        fun = partial(ParamGeneric.fixed_point, **kwargs)
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                hvar = np.eye(nstocks)
                sol = sco.fixed_point(fun, hvar[np.tril_indices(nstocks)])
                hvar[np.tril_indices(nstocks)] = sol
                hvar[np.triu_indices(nstocks, 1)] \
                    = hvar.T[np.triu_indices(nstocks, 1)]
                return hvar
        except RuntimeError:
            # warnings.warn('Could not find stationary varaince!')
            return None

    def get_uvar(self):
        """Unconditional variance matrix regardless of the model.

        Returns
        -------
        (nstocks, nstocks) array
            Unconditional variance amtrix

        """
        return self.find_stationary_var(self.amat, self.bmat, self.cmat)

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

    def uvar_bad(self):
        """Check that unconditional variance is well defined.

        """
        if self.cmat is not None:
            uvar = self.get_uvar()
        else:
            return True
        if uvar is None:
            return True
        elif np.any(np.diag(uvar) <= 0):
            return True
        elif np.any(np.linalg.eigvals(uvar) <= 0):
            return True
        else:
            return False

    def penalty(self):
        """Penalty in the likelihood for bad parameter values.

        """
        nstocks = self.amat.shape[0]
        penalty = np.maximum(np.diag(self.amat - self.bmat),
                             np.zeros(nstocks)).max()
        penalty += np.minimum(np.diag(self.amat), np.zeros(nstocks)).min()**2
        penalty += np.minimum(np.diag(self.bmat), np.zeros(nstocks)).min()**2
        penalty += np.maximum(np.diag(self.amat) - np.ones(nstocks),
                              np.zeros(nstocks)).max()
        penalty += np.maximum(np.diag(self.bmat) - np.ones(nstocks),
                              np.zeros(nstocks)).max()
        penalty += np.maximum(self.constraint() - 1, 0)
        return 1e5 * (np.exp(penalty) - 1)
