#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## import sys
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix/")

import numpy as np
from scipy.interpolate import splprep, splev
import pyrotein as pr
import helix
import GnuplotPy3


## parvals = [0, 0, 2, 1, 0, 1, 5.5, 50 / 180 * np.pi, 10.5, 0.0, 0.0]
## xyzs = helix.parameterize.helixmodel(parvals, 50, [0, 0, 0])
## 
## # Model...
## model_points = helix.tnb.fit_spline(xyzs.T, s = 0, num = 500)

# [[[ DATA ]]]
# Get CA value for visualization
# Read the PDB file...
pdb       = "1gzm"
fl_pdb    =f"{pdb}.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 6...
chain = "A"
nterm = 244
cterm = 277
sample_scale = 1    # 1 means don't over sample;  2 means use double the sample points.
sample_num = (cterm - nterm + 1) * sample_scale

# Obtain coordinates...
xyzs_dict = {}
peptides = ["N", "CA", "C", "O"]
for i in peptides:
    xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

# Model with cubic spline...
model_points = {}
for i in peptides:
    model_points[i] = helix.tnb.fit_spline(xyzs_dict[i].T, num = sample_num)

model_points = model_points["CA"]
xyzs = xyzs_dict["CA"]

# [[[ MEASURE CURVES IN SPACE ]]]
dr1 , dr2 , \
tvec, nvec, bvec, avec, \
curv, tor = helix.tnb.measure_spline(model_points)

if 0:
    gp = GnuplotPy3.GnuplotPy3()
    gp("set term qt")
    gp("set view equal xyz")
    gp("set xlabel 'x'")
    gp("set ylabel 'y'")
    gp("set zlabel 'z'")

    scale = 4
    for i in range(0, len(xyzs) - 1):
        xb, yb, zb = xyzs[i]
        xt = xb + scale * avec.T[i,0]
        yt = yb + scale * avec.T[i,1]
        zt = zb + scale * avec.T[i,2]
        gp(f"set arrow from {xb},{yb},{zb} to {xt},{yt},{zt} linecolor rgb 'red'")

    gp(f"splot \\")
    gp(f"'-' using 1:2:3 with lines linecolor rgb 'blue', \\")
    gp(f"")

    for i in range(len(xyzs)):
        x2, y2, z2 = xyzs[i]
        gp(f"{x2} {y2} {z2}")
    gp(f"e")
    input()


if 1:
    # [[[ EXPORT ]]]
    if 1:
        fl_export = f"example.curve.dat"
        with open(fl_export,'w') as fh:
            for i in range(len(tor[0])):
                fh.write(f"{i} ")
                fh.write(f"{curv[0][i]} ")
                fh.write(f"{tor [0][i]} ")
                fh.write("\n")

    splx, sply, splz = model_points

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

    scale = 4
    for i in range(0, len(xyzs) - 1):
        xb, yb, zb = xyzs[i]
        xt = xb + scale * avec.T[i,0]
        yt = yb + scale * avec.T[i,1]
        zt = zb + scale * avec.T[i,2]
        gp(f"set arrow from {xb},{yb},{zb} to {xt},{yt},{zt} linecolor rgb 'red'")

    gp("splot '-' using 1:2:3 with points pointtype 6 pointsize 1.2 title 'Data', \\")
    gp("      '-' using 1:2:3:4 with lines linewidth 6.0 linecolor pal title 'Spline', \\")
    gp("      '-' using 1:2:3 with lines linecolor rgb 'gray' title 'CA', \\")
    gp("      '-' using 1:2:3:4 with labels notitle, \\")
    gp("")

    x = xyzs.T[0]
    y = xyzs.T[1]
    z = xyzs.T[2]
    for i in range(len(x)):
        gp(f"{x[i]} {y[i]} {z[i]}")
    gp(f"e")

    for i in range(len(splx)):
        x1 = splx[i]
        y1 = sply[i]
        z1 = splz[i]
        t  = tor[0][i]
        gp(f"{x1} {y1} {z1} {t}")
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

    gp("set palette defined \\")
    gp("(-5 '#ffd8cb', 0 '#ff8b67', \\")
    gp("5 '#ff0000')")
    minmax = np.max([np.abs(np.min(curv[0])), np.max(curv[0])])
    gp(f"set cbrange [0:{minmax}]")
    gp("splot '-' using 1:2:3 with points pointtype 6 pointsize 1.2 title 'Data', \\")
    gp("      '-' using 1:2:3:4 with lines linewidth 6.0 linecolor pal title 'Spline', \\")
    gp("      '-' using 1:2:3 with lines linecolor rgb 'gray' title 'CA', \\")
    gp("      '-' using 1:2:3:4 with labels notitle, \\")
    gp("")

    x = xyzs.T[0]
    y = xyzs.T[1]
    z = xyzs.T[2]
    for i in range(len(x)):
        gp(f"{x[i]} {y[i]} {z[i]}")
    gp(f"e")

    for i in range(len(splx)):
        x1 = splx[i]
        y1 = sply[i]
        z1 = splz[i]
        c  = curv[0][i]
        gp(f"{x1} {y1} {z1} {c}")
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
