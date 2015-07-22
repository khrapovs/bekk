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

    def test_init_scalar_false(self):
        """Test scalar restriction without targeting."""

        nstocks = 6
        restriction = 'scalar'
        alpha, beta = .09, .81
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * alpha**.5
        bmat = np.eye(nstocks) * beta**.5
        uvar = np.eye(nstocks)
        # Choose intercept to normalize unconditional variance to one
        craw = uvar - amat.dot(amat) - bmat.dot(bmat)
        cmat = scl.cholesky(craw, 1)
        theta = [[alpha**.5], [beta**.5]]
        theta.append(cmat[np.tril_indices(cmat.shape[0])])
        theta = np.concatenate(theta)

        param = BEKKParams(amat=amat, bmat=bmat, cmat=cmat,
                           restriction=restriction)

        self.assertEqual(param.target, None)
        self.assertEqual(param.restriction, restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_equal(cmat, param.cmat)
        npt.assert_array_equal(uvar, param.unconditional_var())
        npt.assert_array_equal(theta, param.theta)

    def test_init_scalar_true(self):
        """Test scalar restriction with targeting."""

        nstocks = 2
        restriction = 'scalar'
        target = np.eye(nstocks)

        alpha, beta = .09, .81
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * alpha**.5
        bmat = np.eye(nstocks) * beta**.5
        # Choose intercept to normalize unconditional variance to one
        # CC' = H - AHA' - BHB'
        craw = target - amat.dot(target).dot(amat.T) \
            - bmat.dot(target).dot(bmat.T)
        cmat = scl.cholesky(craw, 1)
        theta = [[alpha**.5], [beta**.5]]
        theta = np.concatenate(theta)

        param = BEKKParams(amat=amat, bmat=bmat, target=target,
                           restriction=restriction)

        npt.assert_array_equal(param.target, target)
        self.assertEqual(param.restriction, restriction)

        npt.assert_array_equal(amat, param.amat)
        npt.assert_array_equal(bmat, param.bmat)
        npt.assert_array_almost_equal(cmat, param.cmat)
        npt.assert_array_equal(target, param.unconditional_var())
        npt.assert_array_equal(theta, param.theta)


if __name__ == '__main__':

    ut.main()
