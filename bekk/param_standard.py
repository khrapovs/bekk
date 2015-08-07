#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standadrd parameter class
-------------------------

"""
from __future__ import print_function, division

import numpy as np

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
    from_theta
        Initialize from theta vector
    get_theta
        Convert parameter matrices to 1-dimensional array

    """

    def __init__(self, nstocks=2):
        """Class constructor.

        Parameters
        ----------
        nstocks : int
            Number os stocks in the model

        """
        super(ParamStandard, self).__init__(nstocks)

    @staticmethod
    def get_model():
        """Return model name.

        """
        return 'standard'

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

        if target is None:
            cmat = np.zeros((nstocks, nstocks))
            cmat[np.tril_indices(nstocks)] = theta[2*chunk:]
        else:
            cmat = cls.find_cmat(amat=amat, bmat=bmat, target=target)

        return cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)

    def get_theta(self, restriction='scalar', use_target=True):
        """Convert parameter mratrices to 1-dimensional array.

        Parameters
        ----------
        restriction : str
            Can be
                - 'full'
                - 'diagonal'
                - 'scalar'

        use_target : bool
            Whether to estimate only A and B (True) or C as well (False)

        Returns
        -------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If use_target is True:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If use_target is False:
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

        if not use_target:
            theta.append(self.cmat[np.tril_indices(self.cmat.shape[0])])

        return np.concatenate(theta)
