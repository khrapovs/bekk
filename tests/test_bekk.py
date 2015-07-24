#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for BEKK.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt
import scipy.linalg as scl

from bekk import BEKK, BEKKParams, simulate_bekk
from bekk import filter_var_python, likelihood_python
from bekk.recursion import filter_var
from bekk.likelihood import likelihood_gauss


class BEKKTestCase(ut.TestCase):

    """Test BEKK."""

    def test_simulation(self):
        """Test simulation."""

        nstocks = 6
        nobs = 10
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)

        param = BEKKParams.from_target(amat=amat, bmat=bmat, target=target)

        for distr in ['normal', 'student', 'skewt']:
            innov, hvar = simulate_bekk(param, nobs=nobs, distr=distr)

            self.assertEqual(innov.shape, (nobs, nstocks))
            self.assertEqual(hvar.shape, (nobs, nstocks, nstocks))

    def test_filter_var(self):
        """Test recursions."""

        nstocks = 6
        nobs = 2000
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)
        param = BEKKParams.from_target(amat=amat, bmat=bmat, target=target)
        cmat = param.cmat
        innov, hvar_true = simulate_bekk(param, nobs=nobs, distr='normal')

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        out1 = filter_var_python(hvar, innov, amat, bmat, cmat)

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        out2 = filter_var(hvar, innov, amat, bmat, cmat)

        npt.assert_array_almost_equal(hvar_true, out1)
        npt.assert_array_almost_equal(hvar_true, out2)

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        filter_var_python(hvar, innov, amat, bmat, cmat)
        out1 = hvar.copy()

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        filter_var(hvar, innov, amat, bmat, cmat)
        out2 = hvar.copy()

        npt.assert_array_almost_equal(hvar_true, out1)
        npt.assert_array_almost_equal(hvar_true, out2)

        self.assertIsInstance(out1, np.ndarray)
        self.assertIsInstance(out2, np.ndarray)

    def test_likelihood(self):
        """Test likelihood."""

        nstocks = 6
        nobs = 2000
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)
        param = BEKKParams.from_target(amat=amat, bmat=bmat, target=target)
        cmat = param.cmat

        innov, hvar_true = simulate_bekk(param, nobs=nobs, distr='normal')

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        filter_var(hvar, innov, amat, bmat, cmat)

        out1 = likelihood_python(hvar, innov)
        out2 = likelihood_gauss(hvar, innov)

        self.assertIsInstance(out1, float)
        self.assertIsInstance(out2, float)
        self.assertAlmostEqual(out1, out2)


if __name__ == '__main__':

    ut.main()
