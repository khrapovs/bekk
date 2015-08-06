#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEKK estimation results
=======================

"""
from __future__ import print_function, division

import numpy as np

from .utils import format_time

__all__ = ['BEKKResults']


class BEKKResults(object):

    """Estimation results.

    Attributes
    ----------

    Methods
    -------


    """

    def __init__(self, param_start=None, param_final=None, time_delta=None,
                 opt_out=None):
        """Initialize the class.

        Parameters
        ----------


        """
        self.param_start = param_start
        self.param_final = param_final
        self.time_delta = time_delta
        self.opt_out = opt_out

    def __str__(self):
        """String representation.

        """
        width = 60
        show = '=' * width
        show += '\nOptimization time = %s' % format_time(self.time_delta)
        show += '\nFinal parameters:'
        show += str(self.param_final)
        show += '=' * width
        return show

#        like_start = self.likelihood(self.param_start.theta, kwargs)
#        like_final = self.likelihood(self.param_final.theta, kwargs)
#        # Form the string
#        string = ['\n']
#        string.append('Method = ' + self.method)
#        string.append('Total time (minutes) = %.2f' % self.time_delta)
#        if 'theta_true' in kwargs:
#            string.append('True likelihood = %.2f' % like_true)
#        string.append('Initial likelihood = %.2f' % like_start)
#        string.append('Final likelihood = %.2f' % like_final)
#        string.append('Likelihood difference = %.2f' %
#                      (like_start - like_final))
#        string.append('Success = ' + str(self.opt_out.success))
#        string.append('Message = ' + str(self.opt_out.message))
#        string.append('Iterations = ' + str(self.opt_out.nit))
#        string.extend(self.param_final.log_string())
#        string.append('\nH0 target =\n'
#                      + np.array_str(estimate_h0(self.innov)))
#        # Save results to the log file
#        with open(self.log_file, 'a') as texfile:
#            for istring in string:
#                texfile.write(istring + '\n')
