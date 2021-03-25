#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def remove_nan(xyzs): return xyzs[~np.isnan(xyzs).any(axis = 1)]


def measure_twocores(core_xyz_ref, core_xyz_tar):
    ''' Measure the following aspects of two helical cores.
        - Interhelical distance vector between the centers.
        - Interhelical angle (0-90 degree)
    '''
    # Obtain the centers...
    center_ref = np.nanmean(core_xyz_ref, axis = 0)
    center_tar = np.nanmean(core_xyz_tar, axis = 0)

    # Construct the interhelical distance vector...
    ih_dvec = center_tar - center_ref

    # Calculate the length of interhelical distance vector...
    norm_ih_dvec = np.linalg.norm(ih_dvec)

    # Obtain the helical core vectors...
    core_xyz_ref_nonan = remove_nan(core_xyz_ref)
    core_xyz_tar_nonan = remove_nan(core_xyz_tar)
    core_vec_ref = core_xyz_ref_nonan[-1] - core_xyz_ref_nonan[0]
    core_vec_tar = core_xyz_tar_nonan[-1] - core_xyz_tar_nonan[0]

    # Calculate the interhelical angle...
    core_vec_ref_unit = core_vec_ref / np.linalg.norm(core_vec_ref)
    core_vec_tar_unit = core_vec_tar / np.linalg.norm(core_vec_tar)
    ih_ang = np.arccos( np.dot(core_vec_ref_unit, core_vec_tar_unit) )

    return ih_dvec, norm_ih_dvec, core_vec_ref_unit, core_vec_tar_unit, ih_ang


def calc_interangle(core_xyz_ref, core_xyz_tar):
    ''' Measure the following aspects of two helical cores.
        - Interhelical angle (0-90 degree)
    '''
    # Obtain the helical core vectors...
    core_xyz_ref_nonan = remove_nan(core_xyz_ref)
    core_xyz_tar_nonan = remove_nan(core_xyz_tar)
    core_vec_ref = core_xyz_ref_nonan[-1] - core_xyz_ref_nonan[0]
    core_vec_tar = core_xyz_tar_nonan[-1] - core_xyz_tar_nonan[0]

    # Calculate the interhelical angle...
    core_vec_ref_unit = core_vec_ref / np.linalg.norm(core_vec_ref)
    core_vec_tar_unit = core_vec_tar / np.linalg.norm(core_vec_tar)
    inter_angle = np.arccos( np.dot(core_vec_ref_unit, core_vec_tar_unit) )

    if inter_angle > np.pi / 2.0: inter_angle = np.pi - inter_angle

    return inter_angle


def calc_interdist(core_xyz_ref, core_xyz_tar):
    ''' Measure the following aspects of two helical cores.
        - Interhelical distance vector between the centers.

        Refers to http://geomalgorithms.com/a07-_distance.html for the method.
        Q is ref, P is tar.  
    '''
    # Obtain the helical core vectors...
    core_xyz_ref_nonan = remove_nan(core_xyz_ref)
    core_xyz_tar_nonan = remove_nan(core_xyz_tar)
    core_vec_ref = core_xyz_ref_nonan[-1] - core_xyz_ref_nonan[0]
    core_vec_tar = core_xyz_tar_nonan[-1] - core_xyz_tar_nonan[0]

    # Obtain the starting point...
    q0 = core_xyz_ref_nonan[0]
    p0 = core_xyz_tar_nonan[0]
    w0 = p0 - q0

    # Obtain the directional vector with magnitude...
    v = core_vec_ref
    u = core_vec_tar

    # Math part...
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    de = a * c - b * b    # Denominator
    if de == 0: sc, tc = 0, d / b
    else:       sc, tc = (b * e - c * d) / de, (a * e - b * d) / de

    # Calculate distance...
    wc = w0 + sc * u - tc * v
    inter_dist = np.linalg.norm(wc)

    return inter_dist
