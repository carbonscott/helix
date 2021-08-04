#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import splprep, splev, BSpline

def measure_spline(xyzs, offset = 0):
    ''' Calculate curvature, torsion of a curve, and axial vector under TNB
        frame, also known as Frenet-Serret frame.  
        DOI: 10.1007/s00894-013-1819-7
        URL: https://janakiev.com/blog/framing-parametric-curves/

        Suppose a helix is parameterized by a function h(t), in which t
        traverses a set of integers.  So dt = 1.  Thus, np.gradient is a 
        right way to obtain derivatives like dx/dt, dy/dt, and dz/dt.  
        In fact, as long as it is evenly spaced by m, the gradient should
        return the right value that is scaled by m.  
        (x..., y..., z...)
    '''
    # Obtain derivatives...
    dr1 = np.gradient(xyzs, axis = 1)
    dr2 = np.gradient(dr1,  axis = 1)
    dr3 = np.gradient(dr2,  axis = 1)

    # Obtain T vector...
    norm_dr1 = np.linalg.norm(dr1, axis = 0, keepdims = True)
    tvec = dr1 / norm_dr1

    # Obtain B vector...
    bvec_aux = np.cross(dr1.T, dr2.T).T
    norm_bvec_aux = np.linalg.norm(bvec_aux, axis = 0, keepdims = True)
    bvec = bvec_aux / norm_bvec_aux

    # Obtain N vector...
    nvec_aux = np.cross(bvec.T, tvec.T).T
    norm_nvec_aux = np.linalg.norm(nvec_aux, axis = 0, keepdims = True)
    nvec = nvec_aux / norm_nvec_aux

    # Obtain axial vector...
    # Get norm of db and dt
    db = bvec[:, 1:] - bvec[:, :-1]
    dt = tvec[:, 1:] - tvec[:, :-1]
    norm_db = np.linalg.norm(db, axis = 0, keepdims = True)
    norm_dt = np.linalg.norm(dt, axis = 0, keepdims = True)

    # Get axial vector...
    # Refer to 10.1007/s00894-013-1819-7
    db_div_dt = norm_db / norm_dt
    rt = (1 + db_div_dt)**(-2)
    rb = db_div_dt * rt
    avec  = rt * bvec[:, :-1] + rb * tvec[:, :-1]
    norm_avec = np.linalg.norm(avec, axis = 0, keepdims = True)

    # Obtain curvature...
    curv = norm_bvec_aux / norm_dr1 ** 3

    # Obtain torsion...
    tor_aux = np.nansum(bvec_aux * dr3, axis = 0, keepdims = True)
    tor  = tor_aux / (norm_bvec_aux**2)

    # Obtain angular turn per atom...
    nvec_dot = np.nansum( nvec[:, :-1] * nvec[:,1:], axis = 0 )
    nvec_cosang = np.zeros(len(nvec_dot) + 1)
    nvec_cosang[:-1] = np.arccos(nvec_dot)
    nvec_cosang[-1]  = np.nan

    # Calculate radius and pitch...
    # Refer to 10.1007/s00214-009-0639-4
    denorm = curv ** 2 + tor ** 2
    radius = curv / denorm
    ## rise_per_res  = tor / denorm

    # Calculate rise per residue...
    dxyz = np.array(xyzs)[:, 1:] - np.array(xyzs)[:, :-1]
    norm_dxyz = np.linalg.norm(dxyz, axis = 0, keepdims = True)
    rise_dot = np.nansum( dxyz * avec, axis = 0 )
    rise_cosang = rise_dot / (norm_dxyz * norm_avec)
    rise_per_res = norm_dxyz * rise_cosang

    sli = slice(offset, len(xyzs[0])-offset, 1)
    return tvec        [:, sli], \
           nvec        [:, sli], \
           bvec        [:, sli], \
           avec        [:, sli], \
           curv        [:, sli], \
           tor         [:, sli], \
           nvec_cosang [sli], \
           radius      [:, sli], \
           rise_per_res[:, sli]




def fit_spline(xyzs, s = 0, k = 3, num = 100):
    ''' Fit a spline curve along coordinates saved in xyzs.  

        xyzs has a dimension of (3, ...).  Transpose xyzs if the dimension is
        (..., 3)
    '''
    tck, u = splprep(xyzs, s = s, k = 3)

    # By default, u.min() => 0, u.max() => 1
    u_fine = np.linspace(u.min(), u.max(), num)
    x, y, z = splev(u_fine, tck)

    return x, y, z
