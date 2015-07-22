#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for BEKK.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import scipy.linalg as scl

from bekk import BEKK, BEKKParams, simulate_bekk
from bekk import filter_var_python, likelihood_python
from bekk import filter_var_numba, likelihood_numba
from bekk import recursion, likelihood


class BEKKTestCase(ut.TestCase):

    """Test BEKK."""

    def test_recursion(self):
        """Test recursions."""

        nstocks = 6
        nobs = 2000
        restriction = 'full'
        # A, B, C - n x n matrices
        amat = np.eye(nstocks) * .09**.5
        bmat = np.eye(nstocks) * .9**.5
        # Craw = np.ones((nstocks, nstocks))*.5 + np.eye(nstocks)*.5
        # Choose intercept to normalize unconditional variance to one
        craw = np.eye(nstocks) - amat.dot(amat) - bmat.dot(bmat)
        cmat = scl.cholesky(craw, 1)

        param_true = BEKKParams(a_mat=amat, b_mat=bmat, c_mat=cmat,
                                restriction=restriction, var_target=False)
        innov = simulate_bekk(param_true, nobs=nobs, distr='normal')

        hvar = np.zeros((nobs, nstocks, nstocks), dtype=float)
        hvar[0] = param_true.unconditional_var()

        out1 = recursion(hvar, innov, amat, bmat, cmat)
        out2 = filter_var_python(hvar, innov, amat, bmat, cmat)

        np.testing.assert_array_almost_equal(out1, out2)


if __name__ == '__main__':

    ut.main()
