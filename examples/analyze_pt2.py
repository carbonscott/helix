#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyrotein as pr
from scipy.interpolate import UnivariateSpline
import GnuplotPy3

fl_params = "params.dat"
raw_data = pr.utils.read_file(fl_params, numerical = True)

rmsd_upperlimit = 0.7
data = np.array( [ i for i in raw_data if i[1] < rmsd_upperlimit ] )

# Calculate the spline...
idx = data[:, 0]       # index
x   = data[:, 2 + 0]
y   = data[:, 2 + 1]
z   = data[:, 2 + 2]
splx = UnivariateSpline(idx, x)
sply = UnivariateSpline(idx, y)
splz = UnivariateSpline(idx, z)

gp = GnuplotPy3.GnuplotPy3()
gp("set xlabel 'x'")
gp("set ylabel 'y'")
gp("splot '-' using 1:2:3 with points pointtype 6 title 'Data', \\")
gp("      '-' using 1:2:3 with lines title 'Spline', \\")
gp("")
for i in range(len(idx)):
    gp(f"{x[i]} {y[i]} {z[i]}")
gp(f"e")

for i in range(len(idx)):
    x1 = splx(idx[i])
    y1 = sply(idx[i])
    z1 = splz(idx[i])
    gp(f"{x1} {y1} {z1}")
gp(f"e")

input("Press Enter to exit...")
