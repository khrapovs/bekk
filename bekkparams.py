#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BEKK(1,1) parameter class.

See also
--------
bekk

"""
from __future__ import print_function, division

import numpy as np
import scipy.linalg as sl

from MGARCH.utils import estimate_h0, _bekk_recursion, _product_cc


class BEKKParams(object):
    """Class to hold parameters of the BEKK model in different representations.

    Attributes
    ----------
    a_mat, b_mat, c_mat : (nstocks, nstocks) arrays
        Matrix representations of BEKK parameters
    theta : 1-dimensional array
        Vector of model parameters
    innov : (nobs, ntocks) array
            Return innovations
    restriction : str
        Can be
            - 'full'
            - 'diagonal'
            - 'scalar'
    var_target : bool
        Variance targeting flag. If True, then c_mat is not returned.

    Methods
    -------
    unconditional_var
        Return unconditional variance given parameters
    constraint
        Constraint on parameters for stationarity
    log_string
        Forms the nice string for output

    """

    def __init__(self, restriction=None, var_target=None,
                 innov=None, **kwargs):
        """Class constructor.

        Parameters
        ----------
        restriction : str
            Can be
                - 'full'
                - 'diagonal'
                - 'scalar'
        var_target : bool
            Variance targeting flag. If True, then c_mat is not returned.
        innov : (nobs, ntocks) array
            Return innovations
        kwargs : keyword arguments, optional

        """
        # Defaults:
        self.a_mat, self.b_mat, self.c_mat = None, None, None
        self.innov = innov
        self.theta = None
        self.restriction = restriction
        self.var_target = var_target
#        if not innov is None:
#            self.__init_parameters()
        # Update attributes from kwargs
        self.__dict__.update(kwargs)

        if 'theta' in kwargs:
            self.__convert_theta_to_abc()
        elif 'a_mat' and 'b_mat' in kwargs:
            self.__convert_abc_to_theta()
        else:
            raise TypeError('Not enough arguments to initialize BEKKParams!')

    def __init_parameters(self):
        """Initialize parameter class given innovations only.

        Parameters
        ----------
        innov : (nobs, ntocks) array
            Return innovations

        # TODO : Do I really need this method?

        """
        nstocks = self.innov.shape[1]
        self.a_mat = np.eye(nstocks) * .15
        self.b_mat = np.eye(nstocks) * .95
        self.c_mat = self.__find_c_mat()
        self.__convert_abc_to_theta()

    def __convert_theta_to_abc(self):
        """Convert 1-dimensional array of parameters to matrices.

        Notes
        -----
        a_mat, b_mat, c_mat : (nstocks, nstocks) arrays
            Parameter matrices
        theta : 1d array of parameters
            Length depends on the model restrictions and variance targeting
            If var_targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not var_targeting:
                - + (n-1)*n/2 for parameter C

        """
        nstocks = self.innov.shape[1]
        if self.restriction == 'full':
            chunk = nstocks**2
            sqsize = [nstocks, nstocks]
            self.a_mat = self.theta[:chunk].reshape(sqsize)
            self.b_mat = self.theta[chunk:2*chunk].reshape(sqsize)
        elif self.restriction == 'diagonal':
            chunk = nstocks
            self.a_mat = np.diag(self.theta[:chunk])
            self.b_mat = np.diag(self.theta[chunk:2*chunk])
        elif self.restriction == 'scalar':
            chunk = 1
            self.a_mat = np.eye(nstocks) * self.theta[:chunk]
            self.b_mat = np.eye(nstocks) * self.theta[chunk:2*chunk]
        else:
            raise ValueError('This restriction is not supported!')

        if self.var_target:
            self.c_mat = self.__find_c_mat()
        else:
            self.c_mat = np.zeros((nstocks, nstocks))
            self.c_mat[np.tril_indices(nstocks)] = self.theta[2*chunk:]

    def __convert_abc_to_theta(self):
        """Convert parameter matrices to 1-dimensional array.

        Notes
        -----
        a_mat, b_mat, c_mat : (nstocks, nstocks) arrays
            Parameter matrices
        theta : 1-dimensional array of parameters
            Length depends on the model restrictions and variance targeting
            If var_targeting:
                - 'full' - 2*n**2
                - 'diagonal' - 2*n
                - 'scalar' - 2
            If not var_targeting:
                - + (n-1)*n/2 for parameter c_mat

        """
        if self.restriction == 'full':
            self.theta = [self.a_mat.flatten(), self.b_mat.flatten()]
        elif self.restriction == 'diagonal':
            self.theta = [np.diag(self.a_mat), np.diag(self.b_mat)]
        elif self.restriction == 'scalar':
            self.theta = [[self.a_mat[0, 0]], [self.b_mat[0, 0]]]
        else:
            raise ValueError('This restriction is not supported!')

        if not self.var_target:
            self.theta.append(self.c_mat[np.tril_indices(self.c_mat.shape[0])])
        self.theta = np.concatenate(self.theta)

    def __find_c_mat(self):
        """Solve for C in H = CC' + AHA' + BHB' given A, B, H.

        Parameters
        ----------
        stationary_var : (nstocks, nstocks) arrays
            Stationary variance matrix, H

        """
        stationary_var = self.unconditional_var()
        c_mat_sq = 2*stationary_var - _bekk_recursion(self, stationary_var,
                                                      stationary_var,
                                                      stationary_var)
        # Extract C parameter
        try:
            return sl.cholesky(c_mat_sq, 1)
        except sl.LinAlgError:
            return None

    def find_stationary_var(self):
        """Find fixed point of H = CC' + AHA' + BHB' given A, B, C.

        Returns
        -------
        hvarnew : (nstocks, nstocks) array
            Stationary variance amtrix

        """
        i, norm = 0, 1e3
        hvarold = np.eye(self.a_mat.shape[0])
        cc_mat = _product_cc(self.c_mat)
        while (norm > 1e-3) or (i < 1e3):
            hvarnew = _bekk_recursion(self, cc_mat, hvarold, hvarold)
            norm = np.linalg.norm(hvarnew - hvarold)
            hvarold = hvarnew[:]
            i += 1
        return hvarnew

    def unconditional_var(self):
        """Unconditional variance matrix regardless of the model.

        Returns
        -------
        hvar : (nstocks, nstocks) array
            Unconditional variance amtrix

        """
        if (self.innov is None) or (not self.var_target):
            return self.find_stationary_var()
        else:
            return estimate_h0(self.innov)

    def constraint(self):
        """Compute the largest eigenvalue of BEKK model.

        Returns
        -------
        float
            Largest eigenvalue

        """
        kron_a = np.kron(self.a_mat, self.a_mat)
        kron_b = np.kron(self.b_mat, self.b_mat)
        return np.abs(sl.eigvals(kron_a + kron_b)).max()

    def log_string(self):
        """Create string for log file.

        Returns
        -------
        string : list
            List of strings

        """
        string = []
        string.append('Varinace targeting = ' + str(self.var_target))
        string.append('Model restriction = ' + str(self.restriction))
        string.append('Max eigenvalue = %.4f' % self.constraint())
        string.append('\nA =\n' + np.array_str(self.a_mat))
        string.append('\nB =\n' + np.array_str(self.b_mat))
        string.append('\nC =\n' + np.array_str(self.c_mat))
        string.append('\nH0 estim =\n'
                      + np.array_str(self.unconditional_var()))
        return string