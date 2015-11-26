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

        nobs = 10
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

    def test_sqinnov(self):
        """Test squared returns."""

        nstocks = 2
        innov = np.ones(nstocks)

        innov2 = BEKK.sqinnov(innov)

        npt.assert_array_equal(innov2, np.ones((nstocks, nstocks)))

    def test_forecast(self):
        """Test forecast."""

        nstocks = 2
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        cmat = np.eye(nstocks)
        param = ParamStandard.from_abc(amat=amat, bmat=bmat, cmat=cmat)

        innov = np.ones(nstocks)
        hvar = np.ones((nstocks, nstocks))

        forecast = BEKK.forecast_one(hvar=hvar, innov=innov, param=param)
        exp = cmat.dot(cmat.T)
        exp += amat.dot(innov * innov[:, np.newaxis]).dot(amat.T)
        exp += bmat.dot(hvar).dot(bmat.T)

        self.assertEqual(forecast.shape, (nstocks, nstocks))
        npt.assert_array_equal(forecast, exp)

    def test_loss(self):
        """Test loss function."""

        nstocks = 2
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        cmat = np.eye(nstocks)
        param = ParamStandard.from_abc(amat=amat, bmat=bmat, cmat=cmat)

        innov = np.ones(nstocks)
        hvar = np.ones((nstocks, nstocks))

        pret = BEKK.pret(innov)
        pvar = BEKK.pvar(hvar)
        self.assertIsInstance(pret, float)
        self.assertIsInstance(pvar, float)

        weights = [1, 3]
        pret = BEKK.pret(innov, weights=weights)
        pvar = BEKK.pvar(hvar, weights=weights)

        self.assertIsInstance(pret, float)
        self.assertIsInstance(pvar, float)
        self.assertEqual(pret, 1)
        self.assertEqual(pvar, 1)

        forecast = BEKK.forecast_one(hvar=hvar, innov=innov, param=param)
        proxy = BEKK.sqinnov(innov)

        self.assertEqual(proxy.shape, (nstocks, nstocks))
        self.assertEqual(forecast.shape, (nstocks, nstocks))

        for kind in ['equal', 'minvar']:
            var = BEKK.portf_var(forecast=forecast, alpha=.05, weights=weights)
            self.assertIsInstance(var, float)

            loss_var = BEKK.loss_var(innov=innov[-1], forecast=forecast,
                                     alpha=.05, weights=weights)
            self.assertIsInstance(loss_var, float)

        loss_eucl = BEKK.loss_eucl(forecast=forecast, proxy=proxy)
        loss_frob = BEKK.loss_frob(forecast=forecast, proxy=proxy)
        loss_stein = BEKK.loss_stein(forecast=forecast, proxy=proxy)
        loss_stein2 = BEKK.loss_stein2(forecast=forecast, innov=innov)

        self.assertIsInstance(loss_eucl, float)
        self.assertIsInstance(loss_frob, float)
        self.assertIsInstance(loss_stein, float)
        self.assertIsInstance(loss_stein2, float)

        portf_lscore = BEKK.portf_lscore(forecast=hvar, innov=innov)
        portf_mse = BEKK.portf_mse(forecast=hvar, proxy=proxy)
        portf_qlike = BEKK.portf_qlike(forecast=hvar, proxy=proxy)

        self.assertIsInstance(portf_lscore, float)
        self.assertIsInstance(portf_mse, float)
        self.assertIsInstance(portf_qlike, float)
        self.assertEqual(portf_lscore, .5)
        self.assertEqual(portf_mse, 0)
        self.assertEqual(portf_qlike, 1)

        all_losses = BEKK.all_losses(forecast=forecast, proxy=proxy,
                                     innov=innov)

        self.assertIsInstance(all_losses, dict)


if __name__ == '__main__':

    ut.main()
