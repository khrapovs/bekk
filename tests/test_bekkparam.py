#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for BEKKParams.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt
import scipy.linalg as scl

from bekk import BEKKParams


class BEKKParamsTestCase(ut.TestCase):

    """Test BEKKParams."""

    def test_init(self):
        """Test init."""

        nstocks = 2
        param = BEKKParams(nstocks)
        self.assertIsInstance(param.amat, np.ndarray)
        self.assertIsInstance(param.bmat, np.ndarray)
        self.assertIsInstance(param.cmat, np.ndarray)
        self.assertEqual(param.amat.shape, (nstocks, nstocks))
        self.assertEqual(param.bmat.shape, (nstocks, nstocks))
        self.assertEqual(param.bmat.shape, (nstocks, nstocks))

    def test_find_cmat(self):
        """Test find C matrix."""

        nstocks = 2
        alpha, beta = .09, .81
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * alpha**.5
        bmat = np.eye(nstocks) * beta**.5
        target = np.eye(nstocks)
        # Choose intercept to normalize unconditional variance to one
        cmat1 = BEKKParams.find_cmat(amat=amat, bmat=bmat, target=target)
        ccmat = target - amat.dot(target).dot(amat.T) \
            - bmat.dot(target).dot(bmat.T)
        cmat2 = scl.cholesky(ccmat, 1)

        npt.assert_array_equal(cmat1, cmat2)

    def test_find_stationary_var(self):
        """Test find stationary variance matrix."""

        nstocks = 2
        alpha, beta = .09, .5
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * alpha**.5
        bmat = np.eye(nstocks) * beta**.5
        target = np.eye(nstocks)
        # Choose intercept to normalize unconditional variance to one
        cmat = BEKKParams.find_cmat(amat=amat, bmat=bmat, target=target)
        param = BEKKParams.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        hvar = param.get_uvar()

        npt.assert_array_almost_equal(hvar, target)

        hvar = BEKKParams.find_stationary_var(amat=amat, bmat=bmat, cmat=cmat)

        npt.assert_array_almost_equal(hvar, target)

    def test_from_abc(self):
        """Test init from abc."""

        nstocks = 2
        amat = np.eye(nstocks)
        bmat = np.eye(nstocks)
        cmat = np.eye(nstocks)
        param = BEKKParams.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        nstocks = 2
        alpha, beta = .09, .81
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * alpha**.5
        bmat = np.eye(nstocks) * beta**.5
        target = np.eye(nstocks)
        # Choose intercept to normalize unconditional variance to one
        cmat = BEKKParams.find_cmat(amat=amat, bmat=bmat, target=target)

        param = BEKKParams.from_abc(amat=amat, bmat=bmat, cmat=cmat)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

    def test_from_target(self):
        """Test init from abc."""

        nstocks = 2
        target = np.eye(nstocks)*.5

        param = BEKKParams.from_target(target=target)
        param_default = BEKKParams(nstocks)
        cmat = BEKKParams.find_cmat(amat=param_default.amat,
                                    bmat=param_default.bmat, target=target)
        param_default = BEKKParams.from_abc(amat=param_default.amat,
                                            bmat=param_default.bmat, cmat=cmat)

        npt.assert_array_equal(param.amat, param_default.amat)
        npt.assert_array_equal(param.bmat, param_default.bmat)
        npt.assert_array_equal(param.cmat, cmat)

        amat = np.eye(nstocks)*.1
        bmat = np.eye(nstocks)*.5

        param = BEKKParams.from_target(amat=amat, bmat=bmat, target=target)
        cmat = BEKKParams.find_cmat(amat=amat, bmat=bmat, target=target)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

    def test_spatial(self):
        """Test spatial specification."""

        nstocks = 3
        weights = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        ncat = weights.shape[0]
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        svecs = np.ones((ncat, nstocks)) * gamma**.5
        vvec = np.ones(nstocks)

        param = BEKKParams.from_spatial(avecs=avecs, bvecs=bvecs, svecs=svecs,
                                        vvec=vvec, weights=weights)

        amat = np.diag(avecs[0]) + np.diag(avecs[0]).dot(weights[0])
        bmat = np.diag(bvecs[0]) + np.diag(bvecs[0]).dot(weights[0])
        smat = np.eye(nstocks) - np.diag(svecs[0]).dot(weights[0])
        smat_inv = scl.inv(smat)
        cmat = smat_inv.dot(np.diag(vvec)).dot(smat_inv)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

    def test_theta(self):
        """Test theta."""

        nstocks = 2
        alpha, beta = .09, .81
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * alpha**.5
        bmat = np.eye(nstocks) * beta**.5
        target = np.eye(nstocks)
        cmat = BEKKParams.find_cmat(amat=amat, bmat=bmat, target=target)

        restriction = 'scalar'
        theta = [[alpha**.5], [beta**.5]]
        theta = np.concatenate(theta)

        param = BEKKParams.from_theta(theta=theta, nstocks=nstocks,
                                      target=target, restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'scalar'
        theta = [[alpha**.5], [beta**.5]]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = BEKKParams.from_theta(theta=theta, nstocks=nstocks,
                                      restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'diagonal'
        theta = [np.diag(amat), np.diag(bmat)]
        theta = np.concatenate(theta)

        param = BEKKParams.from_theta(theta=theta, nstocks=nstocks,
                                      target=target, restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'diagonal'
        theta = [np.diag(amat), np.diag(bmat)]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = BEKKParams.from_theta(theta=theta, nstocks=nstocks,
                                      restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'full'
        theta = [amat.flatten(), bmat.flatten()]
        theta = np.concatenate(theta)

        param = BEKKParams.from_theta(theta=theta, nstocks=nstocks,
                                      target=target, restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'full'
        theta = [amat.flatten(), bmat.flatten()]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = BEKKParams.from_theta(theta=theta, nstocks=nstocks,
                                      restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)


if __name__ == '__main__':

    ut.main()
