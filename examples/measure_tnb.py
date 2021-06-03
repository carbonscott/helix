#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix/")    # For Linux
## sys.path.insert(0, "/Users/scott/Dropbox/codes/helix/")   # For mac

import numpy as np
from scipy.interpolate import splprep, splev
import pyrotein as pr
import helix
import GnuplotPy3
import os

if False:
    pdb = "test"
    chain = "A"
    nterm = 0
    sample_num = 100
    parvals = [0, 0, 2, 1, 0, 1, 5.5, 10 / 180 * np.pi, 2.3, 0.0, 0.0]
    xyzs = helix.parameterize.helixmodel(parvals, sample_num, [0, 0, 0])

    # Model...
    model_points = helix.tnb.fit_spline(xyzs.T, s = 100 * sample_num, num = sample_num)

    # [[[ MEASURE CURVES IN SPACE ]]]
    xyzs_model = model_points
    dr1 , dr2 , \
    tvec, nvec, bvec, avec, \
    curv, tor, omega = helix.tnb.measure_spline(xyzs_model)


if True:
    # [[[ DATA ]]]
    # Get CA value for visualization
    # Read the PDB file...
    pdb       = "2hpy"
    fl_pdb    =f"../pdb/{pdb}.pdb"
    atom_list = pr.atom.read(fl_pdb)
    atom_dict = pr.atom.create_lookup_table(atom_list)

    # Select helix 6...
    chain = "A"
    ## nterm = 98
    ## cterm = 108
    nterm = 285
    cterm = 307
    sample_scale = 10
    sample_num = (cterm - nterm + 1) * sample_scale

    # Obtain coordinates...
    xyzs_dict = {}
    peptides = ["N", "CA", "C", "O"]
    for i in peptides:
        xyzs_dict[i] = pr.atom.extract_xyz_by_atom([i], atom_dict, chain, nterm, cterm)

    # Model with cubic spline...
    model_points = {}
    for i in peptides:
        model_points[i] = helix.tnb.fit_spline(xyzs_dict[i].T, num = sample_num)

    xyzs = xyzs_dict["CA"]

    # [[[ MEASURE CURVES IN SPACE ]]]
    xyzs_model = model_points["CA"]
    dr1 , dr2 , \
    tvec, nvec, bvec, avec, \
    curv, tor, omega = helix.tnb.measure_spline(xyzs_model)

if 1:
    scale = 1
    step  = 2
    # [[[ EXPORT ]]]
    if 0:
        fl_export = f"{pdb}_{chain}.dat"
        with open(fl_export,'w') as fh:
            for i in range(len(tor[0])):
                fh.write(f"{nterm + i} ")
                fh.write(f"{tor [0][i]} ")
                fh.write(f"{curv[0][i]} ")
                fh.write(f"{omega[i]} ")
                fh.write("\n")

    splx, sply, splz = xyzs_model

    gp = GnuplotPy3.GnuplotPy3()
    gp("set term qt")
    gp("set view equal xyz")
    gp("set xlabel 'x'")
    gp("set ylabel 'y'")
    gp("set zlabel 'z'")
    gp("set palette defined \\")
    gp("(-5 'red', 0 'seagreen', \\")
    gp("5 'blue')")
    minmax = np.max([np.abs(np.min(tor[0])), np.max(tor[0])])
    gp(f"set cbrange [-{minmax}:{minmax}]")

    # Axial vector
    for i in range(0, len(xyzs_model[0]) - 1, step):
        xb, yb, zb = xyzs_model[0][i], xyzs_model[1][i], xyzs_model[2][i]
        xt = xb + scale * avec.T[i,0]
        yt = yb + scale * avec.T[i,1]
        zt = zb + scale * avec.T[i,2]
        gp(f"set arrow from {xb},{yb},{zb} to {xt},{yt},{zt} linecolor rgb 'green'")

    # T vector
    for i in range(0, len(xyzs_model[0]) - 1, step):
        xb, yb, zb = xyzs_model[0][i], xyzs_model[1][i], xyzs_model[2][i]
        xt = xb + scale * tvec.T[i,0]
        yt = yb + scale * tvec.T[i,1]
        zt = zb + scale * tvec.T[i,2]
        gp(f"set arrow from {xb},{yb},{zb} to {xt},{yt},{zt} linewidth 2.0 linecolor rgb 'red'")

    # N vector
    for i in range(0, len(xyzs_model[0]) - 1, step):
        xb, yb, zb = xyzs_model[0][i], xyzs_model[1][i], xyzs_model[2][i]
        xt = xb + scale * nvec.T[i,0]
        yt = yb + scale * nvec.T[i,1]
        zt = zb + scale * nvec.T[i,2]
        gp(f"set arrow from {xb},{yb},{zb} to {xt},{yt},{zt} linewidth 2.0 linecolor rgb 'blue'")

    # B vector
    for i in range(0, len(xyzs_model[0]) - 1, step):
        xb, yb, zb = xyzs_model[0][i], xyzs_model[1][i], xyzs_model[2][i]
        xt = xb + scale * bvec.T[i,0]
        yt = yb + scale * bvec.T[i,1]
        zt = zb + scale * bvec.T[i,2]
        gp(f"set arrow from {xb},{yb},{zb} to {xt},{yt},{zt} linewidth 2.0 linecolor rgb 'purple'")

    gp("splot '-' using 1:2:3 with points pointtype 7 pointsize 1.0 title 'Data', \\")
    gp("      '-' using 1:2:3 with lines linewidth 2.0 linecolor rgb 'black' title 'Spline', \\")
    gp("      '-' using 1:2:3 with lines linecolor rgb 'gray' title 'CA', \\")
    gp("      '-' using 1:2:3:4 with labels notitle, \\")
    gp("")

    x = xyzs.T[0]
    y = xyzs.T[1]
    z = xyzs.T[2]
    for i in range(len(x)):
        gp(f"{x[i]} {y[i]} {z[i]}")
    gp(f"e")

    for i in range(0, len(splx), step):
        x1 = splx[i]
        y1 = sply[i]
        z1 = splz[i]
        gp(f"{x1} {y1} {z1}")
    gp(f"e")

    for i in range(len(xyzs)):
        x2, y2, z2 = xyzs[i]
        gp(f"{x2} {y2} {z2}")
    gp(f"e")

    for i in range(len(xyzs)):
        x2, y2, z2 = xyzs[i]
        gp(f"{x2} {y2} {z2} {nterm + i}")
    gp(f"e")

    input("Press Enter to exit...")


    # Export data...
    fl_export = os.path.join('.', f"{pdb}_{chain}.dat")
    with open(fl_export,'w') as fh:
        for i in range(len(tor[0])):
            fh.write(f"{nterm + i} ")
            fh.write(f"{tor [0][i]} ")
            fh.write(f"{curv[0][i]} ")
            fh.write(f"{omega[i]} ")
            fh.write("\n")
