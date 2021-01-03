#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize


def point_on_line(t, n, x0): return x0 + t * n


def nearest(t, n, x0, c0):
    ''' Find the nearest point to c0 on the line parameterized by t given
        direction and a passinng point x0.  
    '''
    def dist(_t): 
        line_t = point_on_line(_t, n, x0)
        return np.sum( (line_t - c0) * (line_t - c0) )

    result = optimize.minimize(dist, t)

    return result
