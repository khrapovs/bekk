.. bekk documentation master file, created by
   sphinx-quickstart on Sun Dec 21 16:37:11 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================================
BEKK model simulation and estimation
====================================

This module allows to simulate and estimate the BEKK(1,1) model proposed in [1]_.

The model assumes that demeaned returns :math:`u_t` are conditionally normal:

    .. math::
        u_t = e_t H_t^{1/2},\quad e_t \sim N(0,I),

with variance matrix evolving accrding to the following recursion:

    .. math::
        H_t = CC^\prime + Au_{t-1}u_{t-1}^\prime A^\prime + BH_{t-1}B^\prime.

References
----------
.. [1] Robert F. Engle and Kenneth F. Kroner
    "Multivariate Simultaneous Generalized Arch",
    Econometric Theory, Vol. 11, No. 1 (Mar., 1995), pp. 122-150,
    <http://www.jstor.org/stable/3532933>

Notes
-----

Check this repo for related R library: https://github.com/vst/mgarch/

Alternative optimization library: http://www.pyopt.org/

Contents
--------

.. toctree::
   :maxdepth: 1

   parameters
   estimation
   results
   generate_data