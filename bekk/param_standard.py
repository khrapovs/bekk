#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEKK standadrd parameter class
==============================

"""
from __future__ import print_function, division

import warnings

import numpy as np
import scipy.linalg as sl

from .param_generic import ParamGeneric

__all__ = ['ParamStandard']


class ParamStandard(ParamGeneric):

    """Class to hold parameters of the BEKK model.

    Attributes
    ----------
    amat, bmat, cmat
        Matrix representations of BEKK parameters

    Methods
    -------
    from_abc
        Initialize from A, B, and C arrays
    from_target
        Initialize from A, B, and variance target
    from_spatial
        Initialize from spatial representation
    find_abdmat_spatial
        Initialize amat, bmat, and dmat from spatial representation
    from_theta
        Initialize from theta vector
    from_theta_spatial
        Initialize from theta vector
    get_theta
        Convert parameter matrices to 1-dimensional array
    get_theta_spatial
        Convert parameter matrices to 1-dimensional array
    find_cmat
        Find C matrix given A, B, and H
    find_vvec
        Find v vector given a, b, d, H, and weights
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
        super(ParamStandard, self).__init__(nstocks)

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
            return sl.cholesky(ccmat, 1)
        except sl.LinAlgError:
            warnings.warn('Matrix C is singular!')
            return None

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
                - +(n-1)*n/2 for parameter C
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
                - +(n-1)*n/2 for parameter cmat

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
