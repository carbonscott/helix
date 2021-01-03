#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/pyrotein")

import pyrotein as pr
import numpy as np
import helix
import line

# Read the PDB file...
fl_pdb    = "1gzm.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 3...
chain = "B"
nterm = 149
cterm = 173

# Obtain coordinates...
## peptide = ["N", "CA", "C", "O"]
peptide = ["CA"]
xyzs = pr.atom.extract_xyz(peptide, atom_dict, chain, nterm, cterm)

# Estimate the helix axis vector...
nv0 = helix.estimate_axis(xyzs)

# Estimate the point that the axis of helix passes through...
pv0 = np.mean(xyzs, axis = 0)

# Fitting...
result = helix.protein(xyzs)
params = result.params

# Check...
helix.check_fit(params, xyzs, pv0, nv0)
