#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for BEKK.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from bekk import BEKKResults, ParamStandard, simulate_bekk


class BEKKResultsTestCase(ut.TestCase):

    """Test BEKKResults."""

    def test_weights_equal(self):
        """Test equal weights."""

        nobs, nstocks = 10, 3
        innov = np.ones((nobs, nstocks))

        res = BEKKResults(innov=innov)
        weights = res.weights_equal()

        npt.assert_array_equal(weights, np.ones_like(innov) / nstocks)

    def test_weights_minvar(self):
        """Test minimum variance weights."""

        nstocks = 6
        nobs = 10
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)

        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
        innov, hvar = simulate_bekk(param, nobs=nobs)

        res = BEKKResults(innov=innov, hvar=hvar)
        weights = res.weights_minvar()

        for hvari, wi in zip(hvar, weights):
            hinv = np.linalg.solve(hvari, np.ones(nstocks))
            npt.assert_array_almost_equal(wi, hinv / hinv.sum())

    def test_weights(self):
        """Test weighting function.

        """
        nstocks = 6
        nobs = 10
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)

        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
        innov, hvar = simulate_bekk(param, nobs=nobs)

        res = BEKKResults(innov=innov, hvar=hvar)

        weights = res.weights()

        npt.assert_array_equal(weights, np.ones_like(innov) / nstocks)

        weights = res.weights(kind='equal')

        npt.assert_array_equal(weights, np.ones_like(innov) / nstocks)

        weights = res.weights(kind='minvar')

        for hvari, wi in zip(hvar, weights):
            hinv = np.linalg.solve(hvari, np.ones(nstocks))
            npt.assert_array_almost_equal(wi, hinv / hinv.sum())

    def test_var_ratio(self):
        """Test variance ratio."""

        nstocks = 6
        nobs = 10
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)

        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
        innov, hvar = simulate_bekk(param, nobs=nobs)

        res = BEKKResults(innov=innov, hvar=hvar)

        evar = res.portf_evar()
        rvar = res.portf_rvar()
        vratio = res.loss_var_ratio()
        mvar = res.portf_mvar()

        self.assertEqual(evar.shape, (nobs, ))
        self.assertEqual(rvar.shape, (nobs, ))
        self.assertEqual(vratio.shape, (nobs, ))
        self.assertIsInstance(mvar, float)


if __name__ == '__main__':

    ut.main()
