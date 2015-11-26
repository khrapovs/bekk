#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial parameter class
-----------------------

"""
from __future__ import print_function, division

import itertools

import numpy as np
import scipy.linalg as scl

from .param_generic import ParamGeneric

__all__ = ['ParamSpatial']


class ParamSpatial(ParamGeneric):

    """Class to hold parameters of the BEKK model.

    Attributes
    ----------
    amat, bmat, cmat, avecs, bvecs, dvecs
        Matrix representations of BEKK parameters
    groups
        List of related items
    weights
        Spatial relation matrices

    Methods
    -------
    from_theta
        Initialize from theta vector
    get_theta
        Convert parameter matrices to 1-dimensional array
    get_weight
        Generate weighting matrices given groups

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
        self.dvecs = np.vstack((np.diag(self.cmat), np.zeros(nstocks)))
        self.weights = np.zeros((1, nstocks, nstocks))

    @staticmethod
    def get_model():
        """Return model name.

        """
        return 'spatial'

    @classmethod
    def from_groups(cls, groups=None, target=None, abstart=(.1, .5)):
        """Initialize from groups.

        Parameters
        ----------
        groups : list of lists of tuples
            Encoded groups of items
        target : (nstocks, nstocks) array
            Unconditional variance matrix

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        weights = cls.get_weight(groups=groups)
        ncat, nstocks = weights.shape[:2]
        avecs = np.vstack([np.ones((1, nstocks)) * abstart[0],
                           np.zeros((ncat, nstocks))])
        bvecs = np.vstack([np.ones((1, nstocks)) * abstart[1],
                           np.zeros((ncat, nstocks))])
        dvecs = np.zeros((1 + ncat, nstocks))
        if target is None:
            dvecs[0] = np.ones(nstocks)
        else:
            dvecs[0] = np.diag(target)

        return cls.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                             groups=groups)

    @classmethod
    def from_abdv(cls, avecs=None, bvecs=None, dvecs=None,
                  groups=None):
        """Initialize from spatial representation.

        Parameters
        ----------
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        dvecs : (ncat+1, nstocks) array
            Parameter matrix
        groups : list of lists of tuples
            Encoded groups of items

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        weights = cls.get_weight(groups=groups)
        amat, bmat, dmat = cls.from_vecs_to_mat(avecs=avecs, bvecs=bvecs,
                                                dvecs=dvecs, weights=weights)
        try:
            dmat_inv = scl.inv(dmat)
        except scl.LinAlgError:
            dmat_inv = np.eye(dmat.shape[0])
        ccmat = dmat_inv.dot(np.diag(dvecs[0]**2)).dot(dmat_inv.T)
        try:
            cmat = scl.cholesky(ccmat, 1)
        except scl.LinAlgError:
            # warnings.warn('Matrix C is singular!')
            cmat = np.diag(np.diag(ccmat)**.5)
        param = cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        param.avecs = avecs
        param.bvecs = bvecs
        param.dvecs = dvecs
        param.weights = weights
        param.groups = groups

        return param

    @classmethod
    def from_abcmat(cls, avecs=None, bvecs=None, cmat=None, groups=None):
        """Initialize from spatial representation.

        Parameters
        ----------
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        cmat : (nstocks, nstocks) array
            Lower triangular matrix for teh intercept CC'
        groups : list of lists of tuples
            Encoded groups of items

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        weights = cls.get_weight(groups=groups)
        amat, bmat, dmat = cls.from_vecs_to_mat(avecs=avecs, bvecs=bvecs,
                                                weights=weights)
        param = cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        param.avecs = avecs
        param.bvecs = bvecs
        param.dvecs = None
        param.weights = weights
        param.groups = groups

        return param

    @classmethod
    def from_abt(cls, avecs=None, bvecs=None, target=None, groups=None):
        """Initialize from spatial representation.

        Parameters
        ----------
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        target : (nstocks, nstocks) array
            Variance target matrix
        groups : list of lists of tuples
            Encoded groups of items

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        weights = cls.get_weight(groups=groups)
        amat, bmat, dmat = cls.from_vecs_to_mat(avecs=avecs, bvecs=bvecs,
                                                weights=weights)
        cmat = cls.find_cmat(amat=amat, bmat=bmat, target=target)
        param = cls.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        param.avecs = avecs
        param.bvecs = bvecs
        param.dvecs = None
        param.weights = weights
        param.groups = groups

        return param

    @staticmethod
    def from_vecs_to_mat(avecs=None, bvecs=None, dvecs=None, weights=None):
        """Initialize amat, bmat, and dmat from spatial representation.

        Parameters
        ----------
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        dvecs : (ncat+1, nstocks) array
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
                dmat -= np.diag(dvecs[i+1]).dot(weights[i])

        return amat, bmat, dmat

    @staticmethod
    def vecs_from_theta(theta=None, groups=None):
        """Convert theta to vecs.

        Parameters
        ----------
        theta : 1d array
            Parameter vector
        groups : list of lists of tuples
            Encoded groups of items

        Returns
        -------
        vecs : (ncat+1, nstocks) array
            Spatial representation of parameters

        """
        weights = ParamSpatial.get_weight(groups)
        ncat, nstocks = weights.shape[:2]
        vecs = np.zeros((ncat+1, nstocks))
        vecs[0, :] = theta[:nstocks]
        j = nstocks
        for cat in range(ncat):
            for group in groups[cat]:
                for item in group:
                    vecs[cat+1, item] = theta[j]
                j += 1
        return vecs, theta[j:]

    def theta_from_vecs(self, vecs=None):
        """Convert theta to vecs.

        Parameters
        ----------
        vecs : (ncat+1, nstocks) array
            Spatial representation of parameters

        Returns
        -------
        theta : 1d array
            Parameter vector

        """
        ncat, nstocks = self.weights.shape[:2]
        theta = [vecs[0, :]]
        for cat in range(ncat):
            for group in self.groups[cat]:
                theta.append([vecs[cat+1, group[0]]])
        return np.concatenate(theta)

    @staticmethod
    def ab_from_theta(theta=None, restriction='shomo', groups=None):
        """Initialize A and B spatial from theta vector.

        Parameters
        ----------
        theta : 1d array
            Parameter vector
        groups : list of lists of tuples
            Encoded groups of items
        restriction : str

            Can be
                - 'hetero' (heterogeneous)
                - 'ghomo' (group homogeneous)
                - 'homo' (homogeneous)
                - 'shomo' (scalar homogeneous)

        Returns
        -------
        avecs : (ncat+1, nstocks) array
            Parameter matrix
        bvecs : (ncat+1, nstocks) array
            Parameter matrix
        theta : 1d array
            Parameter vector. What's left after cutting off avecs and bvecs.

        """
        weights = ParamSpatial.get_weight(groups)
        ncat, nstocks = weights.shape[:2]
        if restriction == 'hetero':
            abvecs_size = (ncat+1)*nstocks
            avecs = theta[:abvecs_size].reshape((ncat+1, nstocks))
            theta = theta[abvecs_size:]
            bvecs = theta[:abvecs_size].reshape((ncat+1, nstocks))
            theta = theta[abvecs_size:]
        elif restriction == 'ghomo':
            avecs, theta = ParamSpatial.vecs_from_theta(theta, groups)
            bvecs, theta = ParamSpatial.vecs_from_theta(theta, groups)
        elif restriction == 'homo':
            avecs = np.zeros((ncat+1, nstocks))
            bvecs = np.zeros((ncat+1, nstocks))
            avecs[0] = theta[:nstocks]
            theta = theta[nstocks:]
            avecs[1:] = np.tile(theta[:ncat, np.newaxis], nstocks)
            theta = theta[ncat:]
            bvecs[0] = theta[:nstocks]
            theta = theta[nstocks:]
            bvecs[1:] = np.tile(theta[:ncat, np.newaxis], nstocks)
            theta = theta[ncat:]
        elif restriction == 'shomo':
            avecs = np.tile(theta[:ncat+1, np.newaxis], nstocks)
            theta = theta[ncat+1:]
            bvecs = np.tile(theta[:ncat+1, np.newaxis], nstocks)
            theta = theta[ncat+1:]
        else:
            raise NotImplementedError('Restriction is not implemented!')

        return avecs, bvecs, theta

    @staticmethod
    def d_from_theta(theta=None, restriction='shomo', groups=None):
        """Initialize D and V spatial from theta vector.

        Parameters
        ----------
        theta : 1d array
            Parameter vector
        groups : list of lists of tuples
            Encoded groups of items
        restriction : str

            Can be
                - 'hetero' (heterogeneous)
                - 'ghomo' (group homogeneous)
                - 'homo' (homogeneous)
                - 'shomo' (scalar homogeneous)

        Returns
        -------
        dvecs : (ncat+1, nstocks) array
            Parameter matrix

        """
        weights = ParamSpatial.get_weight(groups)
        ncat, nstocks = weights.shape[:2]

        if restriction == 'hetero':
            dvecs_size = (ncat+1)*nstocks
            dvecs = theta[:dvecs_size]
            dvecs = dvecs.reshape((ncat+1, nstocks))
        elif restriction == 'ghomo':
            dvecs = ParamSpatial.vecs_from_theta(theta, groups)[0]
        elif restriction in ('homo', 'shomo'):
            dvecs = np.zeros((ncat+1, nstocks))
            dvecs[0] = theta[:nstocks]
            theta = theta[nstocks:]
            dvecs[1:] = np.tile(theta[:ncat, np.newaxis], nstocks)
        else:
            raise NotImplementedError('Restriction is not implemented!')

        return dvecs

    @classmethod
    def from_theta(cls, theta=None, groups=None, cfree=False,
                   restriction='shomo', target=None):
        """Initialize from theta vector.

        Parameters
        ----------
        theta : 1d array
            Length depends on the model restrictions and variance targeting
        weights : (ncat, nstocks, nstocks) array
            Weight matrices
        groups : list of lists of tuples
            Encoded groups of items
        cfree : bool
            Whether to leave C matrix free (True) or not (False)
        target : (nstocks, nstocks) array
            Variance target matrix
        restriction : str

            Can be
                - 'hetero' (heterogeneous)
                - 'ghomo' (group homogeneous)
                - 'homo' (homogeneous)
                - 'shomo' (scalar homogeneous)

        Returns
        -------
        param : BEKKParams instance
            BEKK parameters

        """
        weights = cls.get_weight(groups)
        ncat, nstocks = weights.shape[:2]
        avecs, bvecs, theta = cls.ab_from_theta(theta=theta,
                                                restriction=restriction,
                                                groups=groups)

        if (target is None) and (not cfree):
            cmat = None
            dvecs = cls.d_from_theta(theta=theta, groups=groups,
                                     restriction=restriction)
            return cls.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                 groups=groups)
        elif (target is None) and cfree:
            dvecs = None
            cmat = np.zeros((nstocks, nstocks))
            cmat[np.tril_indices(nstocks)] = theta
            return cls.from_abcmat(avecs=avecs, bvecs=bvecs, cmat=cmat,
                                   groups=groups)
        else:
            dvecs = None
            cmat = None
            return cls.from_abt(avecs=avecs, bvecs=bvecs, target=target,
                                groups=groups)

    def get_theta_from_ab(self, restriction='shomo'):
        """Convert parameter matrices A and B to 1-dimensional array.

        Parameters
        ----------
        restriction : str
            Can be
                - 'hetero' (heterogeneous)
                - 'ghomo' (group homogeneous)
                - 'homo' (homogeneous)
                - 'shomo' (scalar homogeneous)

        Returns
        -------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If use_target is True:
                - 'hetero' - 2*n*(m+1)
                - 'ghomo' - 2*(n+m*k)
                - 'homo' - 2*(n+m)
                - 'shomo' - 2*(m+1)

            If use_target is False and cfree is False:
                - 'hetero' - +n*(m+1)
                - 'ghomo' - +(n+m*k)
                - 'homo' - +(n+m)
                - 'shomo' - +(n+m)

            If use_target is False and cfree is True:
                - +n*(n+1)/2

        """
        if restriction == 'hetero':
            theta = [self.avecs.flatten(), self.bvecs.flatten()]
        elif restriction == 'ghomo':
            theta = [self.theta_from_vecs(self.avecs),
                     self.theta_from_vecs(self.bvecs)]
        elif restriction == 'homo':
            theta = [self.avecs[0], self.avecs[1:, 0],
                     self.bvecs[0], self.bvecs[1:, 0]]
        elif restriction == 'shomo':
            theta = [self.avecs[:, 0].flatten(), self.bvecs[:, 0].flatten()]
        else:
            raise NotImplementedError('Restriction is not implemented!')
        return np.concatenate(theta)

    def get_theta(self, restriction='shomo', use_target=False, cfree=False):
        """Convert parameter matrices to 1-dimensional array.

        Parameters
        ----------
        restriction : str

            Can be
                - 'hetero' (heterogeneous)
                - 'ghomo' (group homogeneous)
                - 'homo' (homogeneous)
                - 'shomo' (scalar homogeneous)

        use_target : bool
            Whether to estimate only A and B (True) or C as well (False)
        cfree : bool
            Whether to leave C matrix free (True) or not (False)

        Returns
        -------
        theta : 1d array
            Length depends on the model restrictions and variance targeting

            If use_target is True:
                - 'hetero' - 2*n*(m+1)
                - 'ghomo' - 2*(n+m*k)
                - 'homo' - 2*(n+m)
                - 'shomo' - 2*(m+1)

            If use_target is False and cfree is False:
                - 'hetero' - +n*(m+1)
                - 'ghomo' - +(n+m*k)
                - 'homo' - +(n+m)
                - 'shomo' - +(n+m)

            If use_target is False and cfree is True:
                - +n*(n+1)/2

        """
        theta = self.get_theta_from_ab(restriction)

        if cfree and (not use_target):
            index = np.tril_indices(self.cmat.shape[0])
            theta = np.concatenate([theta, self.cmat[index]])

        elif (not cfree) and (not use_target):

            if self.dvecs is None:
                self.dvecs = np.vstack((np.diag(self.cmat),
                                        np.zeros(self.cmat.shape[0])))

            if restriction == 'hetero':
                theta = np.concatenate([theta, self.dvecs.flatten()])
            elif restriction == 'ghomo':
                theta = np.concatenate([theta,
                                        self.theta_from_vecs(self.dvecs)])
            elif restriction in ('homo', 'shomo'):
                theta = np.concatenate([theta,
                                        self.dvecs[0], self.dvecs[1:, 0]])
            else:
                raise NotImplementedError('Restriction is not implemented!')

        return theta

    @staticmethod
    def get_weight(groups=None):
        """Generate weighting matrices given groups.

        Parameters
        ----------
        groups : list of lists of tuples
            Encoded groups of items

        Returns
        -------
        (ngroups, nitems, nitems) array
            Spatial weights

        Examples
        --------
        >>> print(ParamSpatial.get_weight(groups=[[(0, 1)]]))
        [[[ 0.  1.]
          [ 1.  0.]]]
        >>> print(ParamSpatial.get_weight(groups=[[(0, 1, 2)]]))
        [[[ 0.   0.5  0.5]
          [ 0.5  0.   0.5]
          [ 0.5  0.5  0. ]]]
        >>> print(ParamSpatial.get_weight(groups=[[(0, 1), (2, 3)]]))
        [[[ 0.  1.  0.  0.]
          [ 1.  0.  0.  0.]
          [ 0.  0.  0.  1.]
          [ 0.  0.  1.  0.]]]
        >>> print(ParamSpatial.get_weight(groups=[[(0, 1), (2, 3, 4)]]))
        [[[ 0.   1.   0.   0.   0. ]
          [ 1.   0.   0.   0.   0. ]
          [ 0.   0.   0.   0.5  0.5]
          [ 0.   0.   0.5  0.   0.5]
          [ 0.   0.   0.5  0.5  0. ]]]

        """
        ncat = len(groups)
        nitems = 0
        for group in groups:
            for items in group:
                temp = np.max(items)
                if temp > nitems:
                    nitems = temp
        weight = np.zeros((ncat, nitems+1, nitems+1))
        for i in range(ncat):
            for group in groups[i]:
                for id1, id2 in itertools.product(group, group):
                    if id1 != id2:
                        weight[i, id1, id2] = 1
            norm = weight[i].sum(0)[:, np.newaxis]
            weight[i] /= norm

        return weight
