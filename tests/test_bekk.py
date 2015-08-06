#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for BEKK.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from bekk import BEKK, ParamStandard, ParamSpatial, simulate_bekk
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

        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)

        for distr in ['normal', 'student', 'skewt']:
            innov, hvar = simulate_bekk(param, nobs=nobs, distr=distr)

            self.assertEqual(innov.shape, (nobs, nstocks))
            self.assertEqual(hvar.shape, (nobs, nstocks, nstocks))

    def test_simulation_spatial(self):
        """Test simulation spatial."""

        nstocks = 6
        nobs = 10
        nstocks = 3
        weights = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])
        ncat = weights.shape[0]
        alpha, beta, gamma = .01, .16, .09
        # A, B, C - n x n matrices
        avecs = np.ones((ncat+1, nstocks)) * alpha**.5
        bvecs = np.ones((ncat+1, nstocks)) * beta**.5
        dvecs = np.ones((ncat, nstocks)) * gamma**.5
        vvec = np.ones(nstocks)

        param = ParamSpatial.from_spatial(avecs=avecs, bvecs=bvecs, dvecs=dvecs,
                                        vvec=vvec, weights=weights)

        for distr in ['normal', 'student', 'skewt']:
            innov, hvar = simulate_bekk(param, nobs=nobs, distr=distr)

            self.assertEqual(innov.shape, (nobs, nstocks))
            self.assertEqual(hvar.shape, (nobs, nstocks, nstocks))

    def test_filter_var(self):
        """Test recursions."""

        nstocks = 2
        nobs = 2000
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)
        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)
        cmat = param.cmat

        innov, hvar_true = simulate_bekk(param, nobs=nobs, distr='normal')

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        out1 = filter_var_python(hvar, innov, amat, bmat, cmat)

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param.get_uvar()

        out2 = filter_var(hvar, innov, amat, bmat, cmat)

        idxl = np.tril_indices(nstocks)
        idxu = np.triu_indices(nstocks)

        out2[:, idxu[0], idxu[1]] = out2[:, idxl[0], idxl[1]]

        npt.assert_array_almost_equal(hvar_true, out1)
        npt.assert_array_almost_equal(hvar_true, out2)

    def test_likelihood(self):
        """Test likelihood."""

        nstocks = 2
        nobs = 2000
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        target = np.eye(nstocks)
        param = ParamStandard.from_target(amat=amat, bmat=bmat, target=target)

        innov, hvar = simulate_bekk(param, nobs=nobs, distr='normal')

        out1 = likelihood_python(hvar, innov)
        out2 = likelihood_gauss(hvar, innov)

        self.assertIsInstance(out1, float)
        self.assertIsInstance(out2, float)

        self.assertAlmostEqual(out1, out2)


if __name__ == '__main__':

    ut.main()
