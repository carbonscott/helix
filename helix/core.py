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



