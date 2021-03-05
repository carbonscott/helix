#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix/")    # For Linux
## sys.path.insert(0, "/Users/scott/Dropbox/codes/helix/")   # For mac

import numpy as np
import pyrotein as pr
import helix
from loaddata import load_xlsx
import os

# Specify chains to process...
fl_chain = "chains.comp.xlsx"
lines    = load_xlsx(fl_chain, sheet = "Sheet2")
drc_pdb  = "pdb"
pro      = ".pro"

# Specify the range of atoms from rhodopsin...
nterm = 154
cterm = 367
peptides = ["N", "CA", "C", "O"]

sample_scale = 1    # 1 means don't over sample;  2 means use double the sample points.
sample_num   = (cterm - nterm + 1) * sample_scale

drc_export = "helix.1"

# [[[ DATA ]]]
for i_fl, line in enumerate(lines):
    # Unpack parameters
    _, pdb, chain, species = line[:4]

    # Get CA value for visualization
    # Read the PDB file...
    fl_pdb    = os.path.join(drc_pdb, f"{pdb}{pro}.pdb")
    atom_list = pr.atom.read(fl_pdb)
    atom_dict = pr.atom.create_lookup_table(atom_list)

    # Obtain coordinates...
    xyzs_dict = {}
    for i in peptides:
        xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

    # Model with cubic spline...
    # Initialize model points (place holder arrays)
    model_points = ( np.zeros(sample_num), \
                     np.zeros(sample_num), \
                     np.zeros(sample_num) )

    # Fetch id for nan values
    id_non_nan = ~np.isnan(xyzs_dict["CA"]).any(axis = 1)

    # Find consecutive integers and group them
    id_non_nan_groups = pr.utils.group_consecutive_integer(np.where(id_non_nan)[0])

    # Find spline for each group
    for id_non_nan_group in id_non_nan_groups:
        # Convert to numpy array to facilitate indexing
        id_non_nan_group_npary = np.array(id_non_nan_group)

        # Fetch non nan coordinates
        xyzs_non_nan = xyzs_dict["CA"][id_non_nan_group_npary]
        len_non_nan  = len(id_non_nan_group_npary)
        if not len_non_nan > 3: continue    # Don't fit if data points are not sufficient for a cubic spline

        # Fit spline
        model_points_non_nan = helix.tnb.fit_spline(xyzs_non_nan.T, num = len_non_nan)

        # Return non nan model points to place holder arrays
        for i in range(3):
            model_points[i][id_non_nan_group_npary] = model_points_non_nan[i]

    # [[[ MEASURE CURVES IN SPACE ]]]
    dr1 , dr2 , \
    tvec, nvec, bvec, avec, \
    curv, tor, omega = helix.tnb.measure_spline(model_points)


    # Export data...
    fl_export = os.path.join(drc_export, f"{pdb}_{chain}.dat")
    with open(fl_export,'w') as fh:
        for i in range(len(tor[0])):
            fh.write(f"{nterm + i} ")
            fh.write(f"{tor [0][i]} ")
            fh.write(f"{curv[0][i]} ")
            fh.write(f"{omega[i]} ")
            fh.write("\n")
