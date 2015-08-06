#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ParamStandard.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt
import scipy.linalg as scl

from bekk import ParamStandard


class ParamStandardTestCase(ut.TestCase):

    """Test ParamStandard."""

    def test_init(self):
        """Test init."""

        nstocks = 2
        param = ParamStandard(nstocks)
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
        cmat1 = ParamStandard.find_cmat(amat=amat, bmat=bmat, target=target)
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
        cmat = ParamStandard.find_cmat(amat=amat, bmat=bmat, target=target)
        param = ParamStandard.from_abc(amat=amat, bmat=bmat, cmat=cmat)
        hvar = param.get_uvar()

        npt.assert_array_almost_equal(hvar, target)

        hvar = ParamStandard.find_stationary_var(amat=amat, bmat=bmat, cmat=cmat)

        npt.assert_array_almost_equal(hvar, target)

    def test_from_abc(self):
        """Test init from abc."""

        nstocks = 2
        amat = np.eye(nstocks)
        bmat = np.eye(nstocks)
        cmat = np.eye(nstocks)
        param = ParamStandard.from_abc(amat=amat, bmat=bmat, cmat=cmat)
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
        cmat = ParamStandard.find_cmat(amat=amat, bmat=bmat, target=target)

        param = ParamStandard.from_abc(amat=amat, bmat=bmat, cmat=cmat)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

    def test_from_target(self):
        """Test init from abc."""

        nstocks = 2
        target = np.eye(nstocks)*.5

        param = ParamStandard.from_target(target=target)
        param_default = ParamStandard(nstocks)
        cmat = ParamStandard.find_cmat(amat=param_default.amat,
                                    bmat=param_default.bmat, target=target)
        param_default = ParamStandard.from_abc(amat=param_default.amat,
                                            bmat=param_default.bmat, cmat=cmat)

        npt.assert_array_equal(param.amat, param_default.amat)
        npt.assert_array_equal(param.bmat, param_default.bmat)
        npt.assert_array_equal(param.cmat, cmat)

        amat = np.eye(nstocks)*.1
        bmat = np.eye(nstocks)*.5

        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
        cmat = ParamStandard.find_cmat(amat=amat, bmat=bmat, target=target)

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
        cmat = ParamStandard.find_cmat(amat=amat, bmat=bmat, target=target)

        restriction = 'scalar'
        theta = [[alpha**.5], [beta**.5]]
        theta = np.concatenate(theta)

        param = ParamStandard.from_theta(theta=theta, nstocks=nstocks,
                                      target=target, restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'scalar'
        theta = [[alpha**.5], [beta**.5]]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = ParamStandard.from_theta(theta=theta, nstocks=nstocks,
                                      restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'diagonal'
        theta = [np.diag(amat), np.diag(bmat)]
        theta = np.concatenate(theta)

        param = ParamStandard.from_theta(theta=theta, nstocks=nstocks,
                                      target=target, restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'diagonal'
        theta = [np.diag(amat), np.diag(bmat)]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = ParamStandard.from_theta(theta=theta, nstocks=nstocks,
                                      restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'full'
        theta = [amat.flatten(), bmat.flatten()]
        theta = np.concatenate(theta)

        param = ParamStandard.from_theta(theta=theta, nstocks=nstocks,
                                      target=target, restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)

        restriction = 'full'
        theta = [amat.flatten(), bmat.flatten()]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = ParamStandard.from_theta(theta=theta, nstocks=nstocks,
                                      restriction=restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)


if __name__ == '__main__':

    ut.main()
