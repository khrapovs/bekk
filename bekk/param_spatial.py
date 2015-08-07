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
    amat, bmat, cmat, avecs, bvecs, dvecs, vvec
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
        super(ParamSpatial, self).__init__(nstocks)
        self.avecs = np.vstack((np.diag(self.amat), np.zeros(nstocks)))
        self.bvecs = np.vstack((np.diag(self.bmat), np.zeros(nstocks)))
        self.dvecs = np.zeros((1, nstocks))
        self.vvec = self.find_vvec(avecs=self.avecs, bvecs=self.bvecs,
                                   dvecs=self.dvecs,
                                   weights=np.zeros((1, 3, 3)),
                                   target=np.eye(nstocks))

    @staticmethod
    def get_model():
        """Return model name.

        """
        return 'spatial'

    @classmethod
    def from_spatial(cls, avecs=None, bvecs=None, dvecs=None,
                     vvec=None, cmat=None, target=None, weights=None):
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
        cmat : (nstocks, nstocks) array
            Lower triangular matrix for teh intercept CC'
        target : (nstocks, nstocks) array
            Variance target matrix
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
        if (target is None) and (cmat is None):
            # avecs, bvecs, dvecs, and vvecs are only inputs
            cmat = dmat_inv.dot(np.diag(vvec**2)).dot(dmat_inv)
        elif cmat is None:
            # avecs, bvecs, and target are only inputs
            cmat = cls.find_cmat(amat=amat, bmat=bmat, target=target)

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
        if dvecs is not None:
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
    def from_theta(cls, theta=None, weights=None, cfree=False,
                   restriction='scalar', target=None):
        """Initialize from theta vector.

        Parameters
        ----------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If target is not None:
                - 3*n*(m+1)-n

            If target not None:
                - +n
        weights : (ncat, nstocks, nstocks) array
            Weight matrices
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        target : (nstocks, nstocks) array
            Variance target matrix
        restriction : str
            Can be
                - 'full' = 'diagonal'
                - 'scalar'

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        ncat, nstocks = weights.shape[:2]
        if restriction in ['full', 'diagonal']:
            abvecs_size = (ncat+1)*nstocks
            avecs = theta[:abvecs_size].reshape((ncat+1, nstocks))
            bvecs = theta[abvecs_size:2*abvecs_size].reshape((ncat+1, nstocks))

        elif restriction == 'scalar':
            abvecs_size = ncat+1
            avecs = np.tile(theta[:abvecs_size, np.newaxis], nstocks)
            bvecs = np.tile(theta[abvecs_size:2*abvecs_size, np.newaxis],
                            nstocks)
        else:
            raise NotImplementedError('Restriction is not implemented!')

        if (target is None) and (not cfree):
            cmat = None
            if restriction in ['full', 'diagonal']:
                dvecs_size = ncat*nstocks
                dvecs = theta[2*abvecs_size:2*abvecs_size+dvecs_size]
                dvecs = dvecs.reshape((ncat, nstocks))
                vvec = theta[2*abvecs_size+dvecs_size:]
            elif restriction == 'scalar':
                dvecs_size = ncat
                dvecs = theta[2*abvecs_size:2*abvecs_size+dvecs_size]
                dvecs = np.tile(dvecs[:, np.newaxis], nstocks)
                vvec = np.tile(theta[2*abvecs_size+dvecs_size:], nstocks)
            else:
                raise NotImplementedError('Restriction is not implemented!')

        elif (target is None) and cfree:
            dvecs = None
            vvec = None
            cmat = np.zeros((nstocks, nstocks))
            cmat[np.tril_indices(nstocks)] = theta[2*abvecs_size:]

        else:
            dvecs = None
            vvec = None
            cmat = None

        return cls.from_spatial(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                vvec=vvec, cmat=cmat, target=target,
                                weights=weights)

    def get_theta(self, restriction='scalar', use_target=False, cfree=False):
        """Convert parameter matrices to 1-dimensional array.

        Parameters
        ----------
        use_target : bool
            Whether to estimate only a, b, and d (True) or v as well (False)
        restriction : str
            Can be
                - 'full' = 'diagonal'
                - 'scalar'
        use_target : bool
            Whether to estimate only A and B (True) or C as well (False)
        cfree : bool
            Whether to leave C matrix free (True) or not (False)

        Returns
        -------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If use_target is True:
                - 'full' or 'diagonal' - 2*n*(m+1)
                - 'scalar' - 2*(m+1)
            If use_target is False and cfree is False:
                - 'full' or 'diagonal' - +n
                - 'scalar' - + (m+1)
            If use_target is False and cfree is True:
                - +n*(n+1)/2

        """
        if restriction in ['full', 'diagonal']:
            theta = [self.avecs.flatten(), self.bvecs.flatten()]
        elif restriction == 'scalar':
            theta = [self.avecs[:, 0].flatten(),
                     self.bvecs[:, 0].flatten()]
        else:
            raise NotImplementedError('Restriction is not implemented!')

        if cfree and (not use_target):
            theta.append(self.cmat[np.tril_indices(self.cmat.shape[0])])
        elif (not cfree) and (not use_target):
            if self.dvecs is None:
                shape = (self.avecs.shape[0]-1, self.avecs.shape[1])
                self.dvecs = np.zeros(shape)
            if self.vvec is None:
                self.vvec = np.ones(self.avecs.shape[1])
            if restriction in ['full', 'diagonal']:
                theta.extend([self.dvecs.flatten(), self.vvec])
            elif restriction == 'scalar':
                theta.extend([self.dvecs[:, 0].flatten(),
                              np.array([self.vvec[0]])])
            else:
                raise NotImplementedError('Restriction is not implemented!')

        return np.concatenate(theta)
