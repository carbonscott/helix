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
x = data[:, 0]       # index
y = data[:, 2 + 8]   # rN
spl = UnivariateSpline(x, y, s = 5)

gp = GnuplotPy3.GnuplotPy3()
gp("set xlabel 'x'")
gp("set ylabel 'y'")
gp("plot '-' using 1:2 with points pointtype 6 title 'Data', \\")
gp("     '-' using 1:2 with lines title 'Spline', \\")
gp("")
for i in range(len(x)):
    gp(f"{x[i]} {y[i]}")
gp(f"e")

for i in np.linspace(x[0], x[-1], 500):
    gp(f"{i} {spl(i)}")
gp(f"e")

input("Press Enter to exit...")
