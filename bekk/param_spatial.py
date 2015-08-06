#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial parameter class
-----------------------

"""
from __future__ import print_function, division

import numpy as np
import scipy.linalg as sl

from .param_generic import ParamGeneric

__all__ = ['ParamSpatial']


class ParamSpatial(ParamGeneric):

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
        super(ParamSpatial, self).__init__(nstocks)

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
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        dvecs : (ncat, nstocks) array
            Parameter matrix
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
        cmat = dmat_inv.dot(np.diag(vvec**2)).dot(dmat_inv)
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
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        dvecs : (ncat, nstocks) array
            Parameter matrix
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
    def find_vvec(avecs=None, bvecs=None, dvecs=None,
                  weights=None, target=None):
        r"""Find v vector given a, b, d, H, and weights.
        Solve for diagonal of

        .. math::
            D_{W}\left(H-A_{W}^{0}HA_{W}^{0\prime}
            -B_{W}^{0}HB_{W}^{0\prime}\right)D_{W}^{\prime}

        Parameters
        ----------
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        dvecs : (ncat, nstocks) array
            Parameter matrix
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
        mats = ParamSpatial.find_abdmat_spatial(avecs=avecs, bvecs=bvecs,
                                                dvecs=dvecs, weights=weights)
        amat, bmat, dmat = mats
        ccmat = target - amat.dot(target).dot(amat.T) \
            - bmat.dot(target).dot(bmat.T)
        return np.abs(np.diag(dmat.dot(ccmat).dot(dmat.T)))**.5

    @classmethod
    def from_theta(cls, theta=None, weights=None, target=None):
        """Initialize from theta vector.

        Parameters
        ----------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If var_target:
                - 3*n*(m+1)-n

            If not var_target:
                - +n
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
        """Convert parameter matrices to 1-dimensional array.

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
                - +n

        """
        theta = [self.avecs.flatten(), self.bvecs.flatten(),
                 self.dvecs.flatten()]

        if not var_target:
            theta.append(self.vvec)

        return np.concatenate(theta)
