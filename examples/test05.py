#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

import pyrotein as pr
import numpy as np
import helix

# Read the PDB file...
fl_pdb    = "1gzm.pdb"
## fl_pdb    = "1gzm.alter.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 3...
chain = "A"
nterm = 150
cterm = 167

# Obtain coordinates...
xyzs_dict = {}
peptides = ["N", "CA", "C", "O"]
for i in peptides:
    xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

# Estimate the mean helix axis...
nv0_dict = {}
for i in peptides: nv0_dict[i] = helix.estimate_axis(xyzs_dict[i])
nv0_array = np.array( [ v for v in nv0_dict.values() ] )
nv0 = np.nanmean(nv0_array, axis = 0)

# Estimate the mean position that the axis of helix passes through...
pv0_dict = {}
for i in peptides: pv0_dict[i] = np.nanmean(xyzs_dict[i], axis = 0)
pv0_array = np.array( [ v for v in pv0_dict.values() ] )
pv0 = np.nanmean(pv0_array, axis = 0)

# Fitting...
result = helix.peptide(xyzs_dict)
params = result.params

## params = helix.peptide(xyzs_dict)

# Check...
helix.check_fit_peptide(params, xyzs_dict, pv0, nv0, nterm)
