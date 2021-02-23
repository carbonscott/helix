#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import splprep, splev


def measure_spline(xyzs):
    ''' Calculate curvature, torsion of a curve, and axial vector under TNB frame,
        also known as Frenet-Serret frame.  
        DOI: 10.1007/s00894-013-1819-7
    '''
    # Obtain the derivatives...
    dr1 = np.gradient(xyzs, axis = 1)
    dr2 = np.gradient(dr1,  axis = 1)
    dr3 = np.gradient(dr2,  axis = 1)

    # Obtain T vector...
    norm_dr1 = np.linalg.norm(dr1, axis = 0, keepdims = True)
    tvec = dr1 / norm_dr1

    # Find dT/dt...
    nvec_aux = np.gradient(tvec, axis = 1)

    # Obtain N vector...
    norm_nvec_aux = np.linalg.norm(nvec_aux, axis = 0, keepdims = True)
    nvec = nvec_aux / norm_nvec_aux

    # Obtain B vector...
    bvec_aux = np.cross(dr1.T, dr2.T).T
    norm_bvec_aux = np.linalg.norm(bvec_aux, axis = 0, keepdims = True)
    bvec = bvec_aux / norm_bvec_aux

    # Obtain axial vector...
    # Get norm of db and dt
    db = bvec[:, 1:] - bvec[:, :-1]
    dt = tvec[:, 1:] - tvec[:, :-1]
    norm_db = np.linalg.norm(db, axis = 0, keepdims = True)
    norm_dt = np.linalg.norm(dt, axis = 0, keepdims = True)

    # Get rt and rb
    db_div_dt = norm_db / norm_dt
    rt = (1 + db_div_dt)**(-2)
    rb = db_div_dt * rt
    avec_aux  = rt * bvec[:, :-1] + rb * tvec[:, :-1]
    norm_avec = np.linalg.norm(avec_aux, axis = 0, keepdims = True)
    avec = avec_aux / norm_avec

    # Obtain curvature...
    curv = norm_nvec_aux / norm_dr1

    # Obtain torsion...
    tor_aux = np.sum(bvec_aux * dr3, axis = 0, keepdims = True)
    tor  = tor_aux / (norm_bvec_aux**2)

    return dr1, dr2, tvec, nvec, bvec, avec, curv, tor




def fit_spline(xyzs, s = 0, k = 3, num = 100):
    ''' Fit a spline curve along coordinates saved in xyzs.  

        xyzs has a dimension of (3, ...).  Transpose xyzs if the dimension is
        (..., 3)
    '''
    tck, u = splprep(xyzs, s = 0, k = 3)
    u_fine = np.linspace(0, 1, num)
    x, y, z = splev(u_fine, tck)

    return x, y, z
