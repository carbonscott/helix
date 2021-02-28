#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import helix
import pyrotein as pr
import GnuplotPy3


# [[[ DATA ]]]
# Get CA value for visualization
# Read the PDB file...
pdb       = "1gzm"
fl_pdb    =f"{pdb}.pdb"
atom_list = pr.atom.read(fl_pdb)
atom_dict = pr.atom.create_lookup_table(atom_list)

# Select helix 6...
chain = "A"
nterm = 1
cterm = 322
sample_num = 2000

# Obtain coordinates...
xyzs_dict = {}
peptides = ["N", "CA", "C", "O"]
for i in peptides:
    xyzs_dict[i] = pr.atom.extract_xyz([i], atom_dict, chain, nterm, cterm)

# Model with cubic spline...
model_points = {}
for i in peptides:
    model_points[i] = helix.tnb.fit_spline(xyzs_dict[i].T, num = sample_num)


# [[[ MEASURE CURVES IN SPACE ]]]
dr1  = {}    # First  derivatives
dr2  = {}    # Second derivatives
tvec = {}    # Unit tangent vector
nvec = {}    # Principle unit normal vector
bvec = {}    # Unit binormal vector
avec = {}    # Unit axial vector
curv = {}    # Curvature
tor  = {}    # Torsion
for i in peptides:
    dr1[i], dr2[i], \
    tvec[i], nvec[i], bvec[i], avec[i], \
    curv[i], tor[i] = helix.tnb.measure_spline(model_points[i])


## # [[[ EXPORT ]]]
atm = "CA"
fl_export = f"{pdb}.{atm}.curve.dat"
i_fine = np.linspace(nterm, cterm, sample_num)
with open(fl_export,'w') as fh:
    for i in range(len(tor[atm][0])):
        fh.write(f"{i_fine[i]} ")
        fh.write(f"{curv[atm][0][i]} ")
        fh.write(f"{tor [atm][0][i]} ")
        fh.write("\n")

splx, sply, splz = model_points["CA"]

gp = GnuplotPy3.GnuplotPy3()
gp("set term qt")
gp("set view equal xyz")
gp("set xlabel 'x'")
gp("set ylabel 'y'")
gp("set zlabel 'z'")
gp("set palette defined \\")
gp("(-5 'red', 0 'seagreen', \\")
gp("5 'blue')")
minmax = np.max([np.abs(np.min(tor["CA"][0])), np.max(tor["CA"][0])])
gp(f"set cbrange [-{0.5 * minmax}:{0.5 * minmax}]")
gp("splot '-' using 1:2:3 with points pointtype 6 pointsize 1.2 title 'Data', \\")
gp("      '-' using 1:2:3:4 with lines linewidth 6.0 linecolor pal title 'Spline', \\")
gp("      '-' using 1:2:3 with lines linecolor rgb 'gray' title 'CA', \\")
gp("      '-' using 1:2:3:4 with labels notitle, \\")
gp("")

x = xyzs_dict["CA"].T[0]
y = xyzs_dict["CA"].T[1]
z = xyzs_dict["CA"].T[2]
for i in range(len(x)):
    gp(f"{x[i]} {y[i]} {z[i]}")
gp(f"e")

for i in range(len(splx)):
    x1 = splx[i]
    y1 = sply[i]
    z1 = splz[i]
    t  = tor["CA"][0][i]
    gp(f"{x1} {y1} {z1} {t}")
gp(f"e")

for i in range(len(xyzs_dict["CA"])):
    x2, y2, z2 = xyzs_dict["CA"][i]
    gp(f"{x2} {y2} {z2}")
gp(f"e")

for i in range(len(xyzs_dict["CA"])):
    x2, y2, z2 = xyzs_dict["CA"][i]
    gp(f"{x2} {y2} {z2} {nterm + i}")
gp(f"e")

input("Press Enter to exit...")

gp("set palette defined \\")
gp("(-5 '#ffd8cb', 0 '#ff8b67', \\")
gp("5 '#ff0000')")
minmax = np.max([np.abs(np.min(curv["CA"][0])), np.max(curv["CA"][0])])
gp(f"set cbrange [0:{0.5 * minmax}]")
gp("splot '-' using 1:2:3 with points pointtype 6 pointsize 1.2 title 'Data', \\")
gp("      '-' using 1:2:3:4 with lines linewidth 6.0 linecolor pal title 'Spline', \\")
gp("      '-' using 1:2:3 with lines linecolor rgb 'gray' title 'CA', \\")
gp("      '-' using 1:2:3:4 with labels notitle, \\")
gp("")

x = xyzs_dict["CA"].T[0]
y = xyzs_dict["CA"].T[1]
z = xyzs_dict["CA"].T[2]
for i in range(len(x)):
    gp(f"{x[i]} {y[i]} {z[i]}")
gp(f"e")

for i in range(len(splx)):
    x1 = splx[i]
    y1 = sply[i]
    z1 = splz[i]
    c  = curv["CA"][0][i]
    gp(f"{x1} {y1} {z1} {c}")
gp(f"e")

for i in range(len(xyzs_dict["CA"])):
    x2, y2, z2 = xyzs_dict["CA"][i]
    gp(f"{x2} {y2} {z2}")
gp(f"e")

for i in range(len(xyzs_dict["CA"])):
    x2, y2, z2 = xyzs_dict["CA"][i]
    gp(f"{x2} {y2} {z2} {nterm + i}")
gp(f"e")

input("Press Enter to exit...")
