#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

from loaddata import load_xlsx
import pyrotein as pr
import numpy as np
import helix
import os
import multiprocessing as mp

# Specify chains to process...
fl_chain = "chains.comp.xlsx"
lines    = load_xlsx(fl_chain, sheet = "Sheet1")
drc_pdb  = "pdb"

# Define helix...
TM_segments = {
                "TM1_e"   : [ 35,  50],
                "TM1_m"   : [ 51,  57],
                "TM1_c"   : [ 58,  64],

                "TM2_c1"  : [ 74,  81],
                "TM2_c2"  : [ 81,  88],
                "TM2_pi"  : [ 86,  90],
                "TM2_e"   : [ 92,  100],


                "TM3_e"   : [110, 123],
                "TM3_c"   : [123, 136],

                "TM4_c"   : [150, 167],
                "TM4_m"   : [166, 169],
                "TM4_e"   : [169, 172],

                "TM5_e"   : [202, 208],
                "TM5_pi"  : [207, 212],
                "TM5_c"   : [212, 225],

                "TM6_c1"  : [244, 247],
                "TM6_c2"  : [248, 264],
                "TM6_c"   : [244, 264],
                "TM6_e"   : [265, 276],

                "TM7_e"   : [289, 296],
                "TM7_310" : [296, 300],
                "TM7_c"   : [301, 307],
               }

## for i_fl, line in enumerate(lines):
def parallel(line):
    # Unpack parameters
    _, pdb, chain, _ = line[:4]

    # Read the PDB file...
    fl_pdb    = os.path.join(drc_pdb, f"{pdb}.pdb")
    atom_list = pr.atom.read(fl_pdb)
    atom_dict = pr.atom.create_lookup_table(atom_list)

    # Collect results...
    result_dict = {}
    for seg, (nterm, cterm) in TM_segments.items():
        # Obtain coordinates...
        xyzs_dict = {}
        peptides = ["N", "CA", "C", "O"]
        for i in peptides:
            xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

        # Fitting...
        try: result = helix.parameterize.helix(xyzs_dict, lam = [0.0], report = False)
        except ValueError: pass

        print(f"Fitting {pdb}.{chain}.{seg}: {nterm}...{cterm}")

        # Report...
        params  = result.params
        parvals = helix.parameterize.unpack_params(params)
        res     = helix.parameterize.report_result(result)
        result_dict[f"{seg}"] = res

    drc_out = "helix"
    fl_out  = os.path.join(drc_out, f"{pdb}_{chain}.helixparam.dat")
    with open(fl_out,'w') as fh:
        for i, (seg, res) in enumerate(result_dict.items()):
            fh.write(f"{i:02d}")
            fh.write("    ")
            fh.write(f"{seg:10s}")
            fh.write("    ")
            fh.write( " ".join(res) )
            fh.write("\n")

num_job = 4
if __name__ == "__main__":
    with mp.Pool(num_job) as proc:
        proc.map( parallel, lines )

