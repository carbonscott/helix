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
len_helix = 4

# Go through the helix to obatin parameters data as a function of sequence...
len_peptide = cterm - nterm + 1
params_dict = {}

bindex = 7

# Obtain the helix segment...
xyzs_filtered_dict = {}
for k, v in xyzs_dict.items():
    xyzs_filtered_dict[k] = v[bindex:bindex+len_helix]

# Fitting...
try: result = helix.helix(xyzs_filtered_dict)
except ValueError: pass

print(f"Fitting {bindex + nterm}...{bindex + nterm + len_helix}")

## # Report
## helix.report_params_helix(result.params, title = f"Optimal params:" + \
##                                            f"cost = {result.cost}")
# Save values...
params_dict[bindex] = [result.params, result.cost]





















# [[[ Visualization purpose ]]]
# Estimate the mean helix axis...
nv0_dict = {}
for i in peptides: nv0_dict[i] = helix.estimate_axis(xyzs_filtered_dict[i])
nv0_array = np.array( [ v for v in nv0_dict.values() ] )
nv0 = np.nanmean(nv0_array, axis = 0)

# Estimate the mean position that the axis of helix passes through...
pv0_dict = {}
for i in peptides: pv0_dict[i] = np.nanmean(xyzs_filtered_dict[i], axis = 0)
pv0_array = np.array( [ v for v in pv0_dict.values() ] )
pv0 = np.nanmean(pv0_array, axis = 0)

# Fitting...
params = result.params

# Check...
# Unpack parameters
parvals = helix.unpack_params(params)
helix.check_select_helix(parvals, xyzs_dict, pv0, nv0, nterm, bindex, len_helix)
