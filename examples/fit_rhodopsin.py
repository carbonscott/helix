#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

import pyrotein as pr
import numpy as np
import helix

# Define helix...
TM_segments = { "TM1_e"   : [ 34,  52],
                "TM1_c"   : [ 51,  64],

                "TM2_c"   : [ 73,  85],
                "TM2_pi"  : [ 84,  90],
                "TM2_e"   : [ 91,  98],

                "TM3_e"   : [110, 123],
                "TM3_c"   : [123, 136],

                "TM4_c"   : [150, 168],
                "TM4_e"   : [168, 173],

                "TM5_e"   : [200, 207],
                "TM5_pi"  : [207, 215],
                "TM5_c"   : [216, 226],

                "TM6_c"   : [243, 262],
                "TM6_e"   : [265, 278],

                "TM7_e"   : [289, 296],
                "TM7_310" : [294, 301],
                "TM7_c"   : [301, 307],
               }

# Read the PDB file...
pdb       = "6fk8"
## pdb       = "1gzm"
fl_pdb    =f"../pdb/{pdb}.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix x...
chain = "A"
nterm, cterm = TM_segments["TM5_pi"]

# Obtain coordinates...
xyzs_dict = {}
peptides = ["N", "CA", "C", "O"]
for i in peptides:
    xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

# Fitting...
try: result = helix.parameterize.helix(xyzs_dict, lam = [0.10])
## try: result = helix.parameterize.helix(xyzs_dict, lam = [0.0, 0.00])
except ValueError: pass
## result = helix.parameterize.helix(xyzs_dict, lam = [0.5, 0.5])

print(f"Fitting {nterm}...{cterm}")


# [[[ Visualization purpose ]]]
# Estimate the mean helix axis...
nv0_dict = {}
for i in peptides: nv0_dict[i] = helix.parameterize.estimate_axis(xyzs_dict[i])
nv0_array = np.array( [ v for v in nv0_dict.values() ] )
nv0 = np.nanmean(nv0_array, axis = 0)

# Estimate the mean position that the axis of helix passes through...
pv0_dict = {}
for i in peptides: pv0_dict[i] = np.nanmean(xyzs_dict[i], axis = 0)
pv0_array = np.array( [ v for v in pv0_dict.values() ] )
pv0 = np.nanmean(pv0_array, axis = 0)

# Fitting...
params = result.params

# Check...
# Unpack parameters
parvals = helix.parameterize.unpack_params(params)
helix.parameterize.check_fit_helix(params, xyzs_dict, pv0, nv0, nterm, ["CA"])

## res = helix.parameterize.report_result(result)
## fl_out = f"{pdb}.helixparam.dat"
## with open(fl_out,'w') as fh:
##     fh.write( " ".join(res) )
##     fh.write("\n")
