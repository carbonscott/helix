#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

import pyrotein as pr
import numpy as np
import helix

# Read the PDB file...
fl_pdb    = "1gzm.pdb"
## fl_pdb    = "1gzm.alter.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 3...
chain = "B"
nterm = 150
cterm = 167

# Obtain coordinates...
peptide = ["N", "CA", "C", "O"]
## peptide = ["O"]
for i in peptide:
    xyzs = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

    # Estimate the helix axis vector...
    nv0 = helix.estimate_axis(xyzs)

    # Estimate the point that the axis of helix passes through...
    pv0 = np.nanmean(xyzs, axis = 0)

    # Fitting...
    result = helix.protein(xyzs)
    params = result.params
    parvals = helix.unpack_params(params)

    # Check...
    helix.check_fit(parvals, xyzs, pv0, nv0, nterm)
