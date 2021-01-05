#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

import pyrotein as pr
import numpy as np
import helix

# Read the PDB file...
fl_pdb    = "1gzm.alter.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 3...
chain = "A"
nterm = 154
cterm = 183

# Obtain coordinates...
xyzs_dict = {}
peptides = ["N", "CA", "C", "O"]
for i in peptides:
    xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

# Define the smallest length of helix to fit...
len_segment = 4
step = 1

# Fit the whole helix with the helix segment...
params_dict = helix.whole_helix(xyzs_dict, len_segment, step, nterm, cterm)

# Export...
fl_out = "test08.dat"
helix.export_params_dict(params_dict, fl_out)
