#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ParamSpatial.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt
import scipy.linalg as scl

from bekk import ParamSpatial


class ParamSpatialSpatialTestCase(ut.TestCase):

    """Test spatial parameter class."""

    def test_get_weight(self):
        """Test construction of spatial weights from groups.

        """
        groups = [[(0, 1), (2, 3)]]
        weight = ParamSpatial.get_weight(groups=groups)
        weight_exp = np.zeros((1, 4, 4))
        weight_exp[0, 0, 1] = 1
        weight_exp[0, 1, 0] = 1
        weight_exp[0, 2, 3] = 1
        weight_exp[0, 3, 2] = 1

        npt.assert_almost_equal(weight, weight_exp)

        groups = [[(0, 1, 2)]]
        weight = ParamSpatial.get_weight(groups=groups)
        weight_exp = np.array([[[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]])

        npt.assert_almost_equal(weight, weight_exp)

        groups = [[(0, 1), (2, 3)], [(0, 2), (1, 3)]]
        weight = ParamSpatial.get_weight(groups=groups)
        weight_exp = np.zeros((len(groups), 4, 4))
        weight_exp[0, :2, :2] = np.array([[0, 1], [1, 0]])
        weight_exp[0, 2:, 2:] = np.array([[0, 1], [1, 0]])
        weight_exp[1, 0:3:2, 0:3:2] = np.array([[0, 1], [1, 0]])
        weight_exp[1, 1:4:2, 1:4:2] = np.array([[0, 1], [1, 0]])

        npt.assert_almost_equal(weight, weight_exp)

        groups = [[(0, 1), (2, 3, 4)]]
        weight = ParamSpatial.get_weight(groups=groups)
        weight_exp = np.zeros((len(groups), 5, 5))
        weight_exp[0, :2, :2] = np.array([[0, 1], [1, 0]])
        weight_exp[0, 2:, 2:] = np.array([[[0, .5, .5], [.5, 0, .5],
                                          [.5, .5, 0]]])

        npt.assert_almost_equal(weight, weight_exp)

    def test_init_empty(self):
        """Test spatial specification."""

        nstocks = 3
        param = ParamSpatial(nstocks=nstocks)

        self.assertEqual(param.amat.shape, (nstocks, nstocks))
        self.assertEqual(param.bmat.shape, (nstocks, nstocks))
        self.assertEqual(param.cmat.shape, (nstocks, nstocks))
        self.assertEqual(param.avecs.shape, (2, nstocks))
        self.assertEqual(param.bvecs.shape, (2, nstocks))
        self.assertEqual(param.dvecs.shape, (2, nstocks))
        self.assertEqual(param.weights.shape, (1, nstocks, nstocks))

    def test_from_abdv(self):
        """Test spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        weights = ParamSpatial.get_weight(groups)
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs,
                                       dvecs=dvecs, groups=groups)

        amat = np.diag(avecs[0]) + np.diag(avecs[0]).dot(weights[0])
        bmat = np.diag(bvecs[0]) + np.diag(bvecs[0]).dot(weights[0])
        dmat = np.eye(nstocks) - np.diag(dvecs[1]).dot(weights[0])
        dmat_inv = scl.inv(dmat)
        ccmat = dmat_inv.dot(np.diag(dvecs[0])).dot(dmat_inv)
        cmat = scl.cholesky(ccmat, 1)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)
        npt.assert_array_equal(avecs, param.avecs)
        npt.assert_array_equal(bvecs, param.bvecs)
        npt.assert_array_equal(dvecs, param.dvecs)
        npt.assert_array_equal(weights, param.weights)

        mats = ParamSpatial.from_vecs_to_mat(avecs=avecs, bvecs=bvecs,
                                             dvecs=dvecs, weights=weights)
        amat_new, bmat_new, dmat_new = mats

        npt.assert_array_equal(amat, amat_new)
        npt.assert_array_equal(bmat, bmat_new)
        npt.assert_array_equal(dmat, dmat_new)

    def test_from_abcmat(self):
        """Test spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        weights = ParamSpatial.get_weight(groups)
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        amat = np.diag(avecs[0]) + np.diag(avecs[0]).dot(weights[0])
        bmat = np.diag(bvecs[0]) + np.diag(bvecs[0]).dot(weights[0])
        dmat = np.eye(nstocks) - np.diag(dvecs[1]).dot(weights[0])
        dmat_inv = scl.inv(dmat)
        cmat = dmat_inv.dot(np.diag(dvecs[0])).dot(dmat_inv)

        param = ParamSpatial.from_abcmat(avecs=avecs, bvecs=bvecs, cmat=cmat,
                                         groups=groups)


        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)
        npt.assert_array_equal(avecs, param.avecs)
        npt.assert_array_equal(bvecs, param.bvecs)
        npt.assert_array_equal(None, param.dvecs)
        npt.assert_array_equal(weights, param.weights)

    def test_from_abt(self):
        """Test spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        weights = ParamSpatial.get_weight(groups)
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        amat = np.diag(avecs[0]) + np.diag(avecs[0]).dot(weights[0])
        bmat = np.diag(bvecs[0]) + np.diag(bvecs[0]).dot(weights[0])
        dmat = np.eye(nstocks) - np.diag(dvecs[1]).dot(weights[0])
        dmat_inv = scl.inv(dmat)
        ccmat = dmat_inv.dot(np.diag(dvecs[0])).dot(dmat_inv)
        cmat = scl.cholesky(ccmat, 1)
        target = ParamSpatial.find_stationary_var(amat=amat, bmat=bmat,
                                                  cmat=cmat)
        cmat_new = ParamSpatial.find_cmat(amat=amat, bmat=bmat, target=target)

        npt.assert_array_almost_equal(cmat[np.tril_indices(nstocks)],
                                      cmat_new[np.tril_indices(nstocks)])

        param = ParamSpatial.from_abt(avecs=avecs, bvecs=bvecs, target=target,
                                      groups=groups)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_almost_equal(cmat, param.cmat)
        npt.assert_array_equal(avecs, param.avecs)
        npt.assert_array_equal(bvecs, param.bvecs)
        npt.assert_array_equal(None, param.dvecs)
        npt.assert_array_equal(weights, param.weights)

    def test_get_theta_from_ab(self):
        """Test theta vector for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'hetero'
        theta = np.concatenate([avecs.flatten(), bvecs.flatten()])
        theta_exp = param.get_theta_from_ab(restriction=restriction)

        npt.assert_array_equal(theta, theta_exp)

        restriction = 'ghomo'
        theta = np.concatenate([avecs[0], avecs[1:, :2].flatten(),
                                bvecs[0], bvecs[1:, :2].flatten()])
        theta_exp = param.get_theta_from_ab(restriction=restriction)

        npt.assert_array_equal(theta, theta_exp)

        restriction = 'homo'
        theta = np.concatenate([avecs[0], avecs[1:, 0],
                                bvecs[0], bvecs[1:, 0]])
        theta_exp = param.get_theta_from_ab(restriction=restriction)

        npt.assert_array_equal(theta, theta_exp)

        restriction = 'shomo'
        theta = np.concatenate([avecs[:, 0], bvecs[:, 0]])
        theta_exp = param.get_theta_from_ab(restriction=restriction)

        npt.assert_array_equal(theta, theta_exp)

    def test_get_theta_hetero(self):
        """Test theta vector for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'hetero'
        use_target = True
        theta = np.concatenate([avecs.flatten(), bvecs.flatten()])
        nparams = 2 * nstocks * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        use_target = False
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten()]
        theta = np.concatenate(theta)
        nparams = 3 * nstocks * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        nparams = 2 * nstocks * (1 + ncat) + nstocks * (nstocks + 1) // 2
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

    def test_get_theta_ghomo(self):
        """Test theta vector for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = len(groups)
        alpha = [.01, .02, .03]
        beta = [.04, .05, .06]
        delta = [.07, .08]
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks))
        bvecs = np.ones((ncat+1, nstocks))
        dvecs = np.ones((ncat+1, nstocks))
        avecs[0, :] *= alpha[0]
        avecs[1, :2] *= alpha[1]
        avecs[1, 2:] *= alpha[2]
        bvecs[0, :] *= beta[0]
        bvecs[1, :2] *= beta[1]
        bvecs[1, 2:] *= beta[2]
        dvecs[1, :2] *= delta[0]
        dvecs[1, 2:] *= delta[1]

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'ghomo'
        use_target = True
        theta = [avecs[0], [avecs[1, 0]], [avecs[1, 2]],
                 bvecs[0], [bvecs[1, 0]], [bvecs[1, 2]]]
        theta = np.concatenate(theta)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        npt.assert_array_equal(theta, theta_exp)

        use_target = False
        theta = [avecs[0], [avecs[1, 0]], [avecs[1, 2]],
                 bvecs[0], [bvecs[1, 0]], [bvecs[1, 2]],
                 dvecs[0], [dvecs[1, 0]], [dvecs[1, 2]]]
        theta = np.concatenate(theta)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        npt.assert_array_equal(theta, theta_exp)

        cfree = True
        theta = [avecs[0], [avecs[1, 0]], [avecs[1, 2]],
                 bvecs[0], [bvecs[1, 0]], [bvecs[1, 2]],
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        npt.assert_array_equal(theta, theta_exp)

    def test_get_theta_homo(self):
        """Test theta vector for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'homo'
        use_target = True
        theta = [avecs[0], avecs[1:, 0], bvecs[0], bvecs[1:, 0]]
        theta = np.concatenate(theta)
        nparams = 2 * (nstocks + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        use_target = False
        theta = [avecs[0], avecs[1:, 0], bvecs[0], bvecs[1:, 0],
                 dvecs[0], dvecs[1:, 0]]
        theta = np.concatenate(theta)
        nparams = 3 * (nstocks + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        cfree = True
        theta = [avecs[0], avecs[1:, 0], bvecs[0], bvecs[1:, 0],
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        nparams = 2 * (nstocks + ncat) + nstocks * (nstocks + 1) // 2
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

    def test_get_theta_shomo(self):
        """Test theta vector for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = 1
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'shomo'
        use_target = True
        theta = [avecs[:, 0], bvecs[:, 0]]
        theta = np.concatenate(theta)
        nparams = 2 * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        use_target = False
        theta = [avecs[:, 0], bvecs[:, 0], dvecs[0], dvecs[1:, 0]]
        theta = np.concatenate(theta)
        nparams = nstocks + 3 * ncat + 2
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        cfree = True
        theta = [avecs[:, 0], bvecs[:, 0],
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        nparams = 2 * (1 + ncat) + nstocks * (nstocks + 1) // 2
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

    def test_from_theta_hetero(self):
        """Test init from theta for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = len(groups)
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'hetero'
        target = None
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten()]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(param.cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(param.dvecs, param_new.dvecs)

        target = param.get_uvar()
        theta = [avecs.flatten(), bvecs.flatten()]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        target = None
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten()]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(param.cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(param.dvecs, param_new.dvecs)

        target = param.get_uvar()
        theta = [avecs.flatten(), bvecs.flatten()]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(nstocks)]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            cfree=cfree)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(np.tril(param.cmat), param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(nstocks)]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            cfree=cfree)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(np.tril(param.cmat), param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

    def test_from_theta_shomo(self):
        """Test init from theta for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = len(groups)
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.vstack([np.ones((1, nstocks)),
                           np.ones((ncat, nstocks)) * gamma**.5])

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'shomo'
        target = None
        theta = [avecs[:, 0], bvecs[:, 0], dvecs[0], dvecs[1:, 0]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(param.cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(param.dvecs, param_new.dvecs)

        target = param.get_uvar()
        theta = [avecs[:, 0], bvecs[:, 0]]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        cfree = True
        theta = [avecs[:, 0], bvecs[:, 0],
                 param.cmat[np.tril_indices(nstocks)]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            cfree=cfree)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(np.tril(param.cmat), param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

    def test_from_theta_group(self):
        """Test group init from theta for spatial specification."""

        nstocks = 4
        groups = [[(0, 1), (2, 3)]]
        ncat = len(groups)
        alpha = [.01, .02, .03]
        beta = [.04, .05, .06]
        delta = [.07, .08]
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks))
        bvecs = np.ones((ncat+1, nstocks))
        dvecs = np.ones((ncat+1, nstocks))
        avecs[0, :] *= alpha[0]
        avecs[1, :2] *= alpha[1]
        avecs[1, 2:] *= alpha[2]
        bvecs[0, :] *= beta[0]
        bvecs[1, :2] *= beta[1]
        bvecs[1, 2:] *= beta[2]
        dvecs[1, :2] *= delta[0]
        dvecs[1, 2:] *= delta[1]

        param = ParamSpatial.from_abdv(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                       groups=groups)

        restriction = 'ghomo'
        target = param.get_uvar()
        theta = [np.ones(nstocks) * alpha[0], alpha[1:],
                 np.ones(nstocks) * beta[0],  beta[1:]]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        uvar = param.find_stationary_var(amat=param.amat, bmat=param.bmat,
                                         cmat=cmat)
        npt.assert_array_almost_equal(target, uvar)

        param_new = ParamSpatial.from_theta(theta=theta, groups=groups,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)
        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_almost_equal(cmat, param.cmat)


if __name__ == '__main__':

    ut.main()
