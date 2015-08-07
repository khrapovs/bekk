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

    """Test spatial ParamSpatial."""

    def test_init_empty(self):
        """Test spatial specification."""

        nstocks = 3
        param = ParamSpatial(nstocks=nstocks)

        self.assertEqual(param.amat.shape, (nstocks, nstocks))
        self.assertEqual(param.bmat.shape, (nstocks, nstocks))
        self.assertEqual(param.cmat.shape, (nstocks, nstocks))
        self.assertEqual(param.avecs.shape, (2, nstocks))
        self.assertEqual(param.bvecs.shape, (2, nstocks))
        self.assertEqual(param.dvecs.shape, (1, nstocks))
        self.assertEqual(param.vvec.shape, (nstocks, ))

    def test_init_spatial(self):
        """Test spatial specification."""

        nstocks = 3
        weights = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        ncat = weights.shape[0]
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.ones((ncat, nstocks)) * gamma**.5
        vvec = np.ones(nstocks)

        param = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs,
                                          dvecs=dvecs, vvec=vvec,
                                          weights=weights)

        amat = np.diag(avecs[0]) + np.diag(avecs[0]).dot(weights[0])
        bmat = np.diag(bvecs[0]) + np.diag(bvecs[0]).dot(weights[0])
        dmat = np.eye(nstocks) - np.diag(dvecs[0]).dot(weights[0])
        dmat_inv = scl.inv(dmat)
        cmat = dmat_inv.dot(np.diag(vvec)).dot(dmat_inv)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)
        npt.assert_array_equal(avecs, param.avecs)
        npt.assert_array_equal(bvecs, param.bvecs)
        npt.assert_array_equal(dvecs, param.dvecs)
        npt.assert_array_equal(vvec, param.vvec)
        npt.assert_array_equal(weights, param.weights)

        mats = ParamSpatial.find_abdmat_spatial(avecs=avecs, bvecs=bvecs,
                                                dvecs=dvecs, weights=weights)
        amat_new, bmat_new, dmat_new = mats

        npt.assert_array_equal(amat, amat_new)
        npt.assert_array_equal(bmat, bmat_new)
        npt.assert_array_equal(dmat, dmat_new)

    def test_get_theta(self):
        """Test theta vector for spatial specification."""

        nstocks = 3
        weights = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        ncat = weights.shape[0]
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.ones((ncat, nstocks)) * gamma**.5
        vvec = np.ones(nstocks)

        param = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs,
                                          dvecs=dvecs, vvec=vvec,
                                          weights=weights)

        restriction = 'full'
        use_target = True
        theta = [avecs.flatten(), bvecs.flatten()]
        theta = np.concatenate(theta)
        nparams = 2 * nstocks * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'full'
        use_target = False
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten(), vvec]
        theta = np.concatenate(theta)
        nparams = 3 * nstocks * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'diagonal'
        use_target = True
        theta = [avecs.flatten(), bvecs.flatten()]
        theta = np.concatenate(theta)
        nparams = 2 * nstocks * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'diagonal'
        use_target = False
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten(), vvec]
        theta = np.concatenate(theta)
        nparams = 3 * nstocks * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'scalar'
        use_target = True
        theta = [avecs[:, 0], bvecs[:, 0]]
        theta = np.concatenate(theta)
        nparams = 2 * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'scalar'
        use_target = False
        theta = [avecs[:, 0], bvecs[:, 0], dvecs[:, 0], np.array([vvec[0]])]
        theta = np.concatenate(theta)
        nparams = 3 * (1 + ncat)
        theta_exp = param.get_theta(restriction=restriction,
                                    use_target=use_target)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'full'
        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        nparams = 2 * nstocks * (1 + ncat) + nstocks * (nstocks + 1) // 2
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'diagonal'
        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        nparams = 2 * nstocks * (1 + ncat) + nstocks * (nstocks + 1) // 2
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

        restriction = 'scalar'
        cfree = True
        theta = [avecs[:, 0], bvecs[:, 0],
                 param.cmat[np.tril_indices(param.cmat.shape[0])]]
        theta = np.concatenate(theta)
        nparams = 2 * (1 + ncat) + nstocks * (nstocks + 1) // 2
        theta_exp = param.get_theta(restriction=restriction, cfree=cfree)

        self.assertEqual(nparams, theta_exp.size)
        npt.assert_array_equal(theta, theta_exp)

    def test_find_vvec(self):
        """Test finding v vector given variance target."""

        nstocks = 3
        weights = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        ncat = weights.shape[0]
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.ones((ncat, nstocks)) * gamma**.5
        target = np.eye(nstocks)

        vvec = ParamSpatial.find_vvec(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                    weights=weights, target=target)

        self.assertEqual(vvec.shape, (nstocks,))

        vvec = np.ones(nstocks)
        param = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs,
                                          dvecs=dvecs, vvec=vvec,
                                          weights=weights)
        target = param.get_uvar()
        vvec_new = ParamSpatial.find_vvec(avecs=avecs, bvecs=bvecs,
                                          dvecs=dvecs, weights=weights,
                                          target=target)
        # TODO : The test fails. Find out why.
        # npt.assert_array_equal(vvec, vvec_new)

    def test_from_theta(self):
        """Test init from theta for spatial specification."""

        nstocks = 3
        weights = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        ncat = weights.shape[0]
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.ones((ncat, nstocks)) * gamma**.5
        vvec = np.ones(nstocks)

        param = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs,
                                          dvecs=dvecs, vvec=vvec,
                                          weights=weights)

        restriction = 'full'
        target = None
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten(), vvec]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(param.cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(param.dvecs, param_new.dvecs)

        restriction = 'full'
        target = param.get_uvar()
        theta = [avecs.flatten(), bvecs.flatten()]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        restriction = 'diagonal'
        target = None
        theta = [avecs.flatten(), bvecs.flatten(), dvecs.flatten(), vvec]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(param.cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(param.dvecs, param_new.dvecs)

        restriction = 'diagonal'
        target = param.get_uvar()
        theta = [avecs.flatten(), bvecs.flatten()]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        restriction = 'scalar'
        target = None
        theta = [avecs[:, 0], bvecs[:, 0], dvecs[:, 0], np.array([vvec[0]])]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(param.cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(param.dvecs, param_new.dvecs)

        restriction = 'scalar'
        target = param.get_uvar()
        theta = [avecs[:, 0], bvecs[:, 0]]
        theta = np.concatenate(theta)
        cmat = param.find_cmat(amat=param.amat, bmat=param.bmat, target=target)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            target=target)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(cmat, param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        restriction = 'full'
        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(nstocks)]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            cfree=cfree)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(np.tril(param.cmat), param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        restriction = 'diagonal'
        cfree = True
        theta = [avecs.flatten(), bvecs.flatten(),
                 param.cmat[np.tril_indices(nstocks)]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            cfree=cfree)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(np.tril(param.cmat), param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)

        restriction = 'scalar'
        cfree = True
        theta = [avecs[:, 0], bvecs[:, 0],
                 param.cmat[np.tril_indices(nstocks)]]
        theta = np.concatenate(theta)
        param_new = ParamSpatial.from_theta(theta=theta, weights=weights,
                                            restriction=restriction,
                                            cfree=cfree)

        npt.assert_array_equal(param.amat, param_new.amat)
        npt.assert_array_equal(param.bmat, param_new.bmat)
        npt.assert_array_equal(np.tril(param.cmat), param_new.cmat)
        npt.assert_array_equal(param.avecs, param_new.avecs)
        npt.assert_array_equal(param.bvecs, param_new.bvecs)
        npt.assert_array_equal(None, param_new.dvecs)


if __name__ == '__main__':

    ut.main()
