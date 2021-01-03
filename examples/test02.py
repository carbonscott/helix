#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/pyrotein")

import pyrotein as pr
import numpy as np
import helix

# Read the PDB file...
fl_pdb    = "1gzm.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 3...
chain = "B"
nterm = 149
cterm = 183

# Obtain coordinates...
## peptide = ["N", "CA", "C", "O"]
peptide = ["CA"]
xyzs = pr.atom.extract_xyz(peptide, atom_dict, chain, nterm, cterm)

# Find the good segment...
helixlen = 20
bindex, result = helix.protein_fit_by_length(xyzs, helixlen)
helix.report_params(result.params, title = f"Optimal params:" + \
                                           f"cost = {result.cost}")
print(f"Select residues: [{nterm + bindex}, {nterm + bindex + helixlen})")
xyzs_sel = xyzs[bindex:bindex+helixlen]

# Estimate the helix axis vector...
nv0 = helix.estimate_axis(xyzs_sel)

# Estimate the point that the axis of helix passes through...
pv0 = np.mean(xyzs_sel, axis = 0)

# Fitting...
## result = helix.protein(xyzs)
params = result.params

# Check...
## helix.check_fit(params, xyzs[bindex:bindex+helixlen], pv0, nv0)
helix.check_select(params, xyzs, pv0, nv0, nterm, bindex, helixlen)
