#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

import pyrotein as pr
import numpy as np
import helix
import os


# Define helix...
seg_dict = { ## "TM1_e"   : [ 35,  50],
             ## "TM1_m"   : [ 51,  56],
             ## "TM1_c"   : [ 57,  63],

             ## "TM2_c1"  : [ 74,  81],
             ## "TM2_c2"  : [ 81,  88],
             ## "TM2_c"   : [ 73,  86],
             ## "TM2_pi"  : [ 86,  91],
             ## "TM2_e"   : [ 92,  98],

             ## "TM3_e"   : [110, 123],
             ## "TM3_m"   : [123, 127],
             ## "TM3_c"   : [123, 136],
             ## "TM3"     : [110, 136],

             ## "TM4_c"   : [150, 167],
             ## "TM4_m"   : [166, 169],
             ## "TM4_e"   : [169, 172],

             ## "TM5_e"   : [202, 208],
             ## "TM5_pi"  : [208, 212],
             ## "TM5_c"   : [212, 225],

             ## "TM6_c1"  : [244, 247],
             ## "TM6_c2"  : [248, 264],
             "TM6_c"   : [244, 264],
             "TM6_e"   : [265, 276],

             "TM7_e"   : [289, 296],
             "TM7_310" : [296, 300],
             "TM7_310.zhong" : [294, 301],
             "TM7_c"   : [302, 307],
         }


# Read helixparam from files...
drc_helix  = "helix"
pdb, chain = "6pel", "A"
id_param   = f"{pdb}_{chain}.align"
fl_param   = os.path.join(drc_helix, f"{id_param}.helixparam.dat")
data       = pr.utils.read_file(fl_param)

# Form a dictionary...
param_dict = {}
for line in data:
    seg = line[1]
    dat = [ float(i) for i in line[2:] ]
    param_dict[seg] = dat


# Metadata about PDB
backbone  = ["N", "CA", "C", "O"]
drc_pdb   = "pdb"
fl_pdb    = os.path.join(drc_pdb, f"{id_param}.pdb")
atoms_pdb = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atoms_pdb)


# [[[ User choice: TM6_e only ]]]
# Select the segment...
seg = "TM6_e"
core_xyzs_dict = {}
helix_xyzs_dict = {}
for seg, _ in seg_dict.items():
    nterm, cterm = seg_dict[seg]

    # Obtain coordinates...
    xyzs = pr.atom.extract_xyz(["CA"], atom_dict, chain, nterm, cterm)

    # [[[ Helix visualization ]]]
    # Find the length of the helix...
    len_helix = xyzs.shape[0]

    # Filter out nan values in xyzs and fetch the coordinate from the first atom...
    nonan_xyzs = xyzs[~np.isnan(xyzs).any(axis = 1)]
    first_xyz  = nonan_xyzs[0]

    # Model the helix core...
    if not seg in param_dict: continue
    parvals     = param_dict[seg]
    parval_dict = helix.parameterize.form_parval_dict(parvals)
    core_xyzs   = helix.parameterize.helixcore(parval_dict["CA"], len_helix)
    helix_xyzs  = helix.parameterize.helixmodel(parval_dict["CA"], len_helix, first_xyz)

    core_xyzs_dict[seg]  = core_xyzs
    helix_xyzs_dict[seg] = helix_xyzs


# Visualize
py_bs = os.path.basename(__file__)[:-3]
import GnuplotPy3
gp = GnuplotPy3.GnuplotPy3()
## gp("set terminal postscript eps  size 3.5, 2.62 \\")
## gp("                             enhanced color \\")
## gp("                             font 'Helvetica,14' \\")
## gp("                             linewidth 2")
## gp(f"set output '{py_bs}.eps'")
gp("set term qt")
gp("set view equal xyz")

gp("splot \\") 
gp("'-' using 1:2:3 with points pointtype 7 pointsize 0.5 linecolor 'black' title 'Core', \\")
gp("'-' using 1:2:3 with linespoints pointtype 7 title 'Helix', \\")
gp("")

for seg, _ in seg_dict.items():
    if not seg in param_dict: continue
    for x, y, z in core_xyzs_dict[seg]:
        gp(f"{x} {y} {z}")
    gp(" ")
gp("e")

for seg, _ in seg_dict.items():
    if not seg in param_dict: continue
    for x, y, z in helix_xyzs_dict[seg]:
        gp(f"{x} {y} {z}")
    gp(" ")
gp("e")

input("")
gp("exit")
