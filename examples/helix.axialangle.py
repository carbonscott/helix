#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## import sys
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix/")    # For Linux
## sys.path.insert(0, "/Users/scott/Dropbox/codes/helix/")   # For mac

import numpy as np
from scipy.interpolate import splprep, splev
import pyrotein as pr
import helix
from loaddata import load_xlsx
import GnuplotPy3
import os


def helical_label(v1, v2, v3):
    ''' Order: 3/10, alpha, pi.  
    '''
    return [
        f"set arrow from graph 0, first {v1} to graph 1, first {v1} nohead dashtype 2 linewidth 2.0 linecolor rgb 'gray'",
        f"set arrow from graph 0,first {v2} to graph 1, first {v2} nohead dashtype 2 linewidth 2.0 linecolor rgb 'gray'",
        f"set arrow from graph 0,first {v3} to graph 1, first {v3} nohead dashtype 2 linewidth 2.0 linecolor rgb 'gray'",
        f"set label '3_{{10}}'      at graph 0.5,first {v1} front",
        f"set label '{{/Symbol a}}' at graph 0.5,first {v2} front",
        f"set label '{{/Symbol p}}' at graph 0.5,first {v3} front", 
    ]

def report(theta, r_list, p_list, omega, path_eps):
    gp = GnuplotPy3.GnuplotPy3()
    gp("set terminal postscript eps  size 3.5, 5.2 \\")
    gp("                             enhanced color \\")
    gp("                             font 'Helvetica,14' \\")
    gp("                             linewidth 1")
    gp(f"set output '{path_eps}'")
    gp("set encoding utf8")
    gp(f"unset key")

    gp(f"set origin 0.0, 0.0")
    gp(f"set size 1, 1")
    gp("unset bmargin")
    gp("unset tmargin")
    gp("unset lmargin")
    gp("unset rmargin")
    gp(f"set multiplot layout 5,1")

    # PLOT 1: Cumulative axial angle
    gp("unset arrow")
    gp("unset label")
    gp(f"unset xrange")
    gp(f"unset yrange")
    gp("unset xtics")
    gp("unset ytics")
    gp(f"unset logscale")
    gp("set tmargin 2")
    gp("set bmargin 0")
    gp("set lmargin at screen 0.15")
    gp("set rmargin at screen 0.95")
    lbl_dict = { i * sample_per_rise : v for i, v in enumerate(range(nterm, cterm+1)) }
    lbl_min, lbl_max = min(lbl_dict.keys()), max(lbl_dict.keys())
    gp(f"set xrange [{lbl_min} : {lbl_max}]")
    gp(f"set xtics 2")
    for i, (k, v) in enumerate(lbl_dict.items()):
        ## if not k % 2: gp(f"set xtics add ('{v}' {k}) rotate by 90 right")
        if not k % 2: gp(f"set xtics add ('' {k}) rotate by 90 right")
    gp("set grid xtics")
    gp("unset yrange")
    gp("set yrange [0:45]")
    gp("set ytics auto")
    gp("set ylabel 'Cumulative axial angle ({\260})'")

    gp("plot \\")
    gp(f"'-' using 1:2 with      points pointtype 7 linecolor rgb 'blue',\\")
    gp("")
    for i, v in enumerate(theta.reshape(-1)):
        gp(f"{i} {v}")
    gp("e")

    # PLOT 2: Radius
    gp("unset title")
    gp("unset arrow")
    gp("unset label")
    gp("set tmargin 2")
    gp("set bmargin 0")
    gp("set lmargin at screen 0.15")
    gp("set rmargin at screen 0.95")
    gp("set ylabel 'Radius ({\305})'")
    gp("set ytics 0.4")
    for i in helical_label(1.868, 2.274, 2.714):
        gp(f"{i}")
    gp("unset yrange")
    gp("set yrange [1.4:3.2]")
    gp("plot \\")
    gp(f"'-' using 1:2 with      points pointtype 7 linecolor rgb 'blue',\\")
    gp("")
    for i, v in enumerate(r_list.reshape(-1)):
        gp(f"{i} {v}")
    gp("e")

    # PLOT 3: Rise
    gp("unset title")
    gp("unset arrow")
    gp("unset label")
    gp("set tmargin 2")
    gp("set bmargin 0")
    gp("set lmargin at screen 0.15")
    gp("set rmargin at screen 0.95")
    gp("set ylabel 'Rise ({\305})'")
    for i, (k, v) in enumerate(lbl_dict.items()):
        if not k % 2: gp(f"set xtics add ('' {k}) rotate by 90 right")
    ## gp("set xlabel 'Residues'")
    gp("unset yrange")
    gp("set yrange [0.8:2.2]")
    gp("set ytics auto")
    for i in helical_label(1.955, 1.517, 0.979):
        gp(f"{i}")

    gp("plot \\")
    gp(f"'-' using 1:2 with      points pointtype 7 linecolor rgb 'blue',\\")
    gp("")
    for i, v in enumerate(p_list.reshape(-1)):
        gp(f"{i} {v}")
    gp("e")

    # PLOT 4: Turn per residue
    gp("unset title")
    gp("unset arrow")
    gp("unset label")
    gp("set tmargin 2")
    gp("set bmargin 0")
    gp("set lmargin at screen 0.15")
    gp("set rmargin at screen 0.95")
    gp("set ylabel 'Turn per residue ({\260})'")
    for i, (k, v) in enumerate(lbl_dict.items()):
        if not k % 2: gp(f"set xtics add ('{v}' {k}) rotate by 90 right")
    gp(f"set xlabel '{pdb_prefix} Residues'")
    gp("unset yrange")
    gp("set yrange")
    gp("set yrange [80:130]")
    gp("set ytics auto")
    for i in helical_label(121.5, 100.1, 85.2):
        gp(f"{i}")

    gp("plot \\")
    gp(f"'-' using 1:2 with      points pointtype 7 linecolor rgb 'blue',\\")
    gp("")
    for i, v in enumerate(omega.reshape(-1)):
        v *= 180 / np.pi
        gp(f"{i} {v}")
    gp("e")

    gp("unset multiplot")
    gp("exit")


# [[[ ANALYZE PDB ENTRIES ]]]
# Specify chains to process...
fl_chain = "chains.comp.xlsx"
lines    = load_xlsx(fl_chain, sheet = "DesKC", splitchain = True)

# Select residues...
nterm = 170
cterm = 206
## cterm = 200
sample_per_rise = 1
num_turn = (cterm - nterm + 1)

# Consider offset...
offterm = 4
nterm_aux = nterm - offterm
cterm_aux = cterm + offterm
num_turn += offterm * 2

# The range to highlight...
anots = [
]

# [[[ DATA ]]]
for i_fl, line in enumerate(lines[:]):
    # Unpack parameters
    _, pdb, chain, _ = line[:4]

    # Get CA value for visualization
    # Read the PDB file...
    ## pdb        = "5iun"
    ## chain      = "A"
    pdb_prefix = f"{pdb}.{chain}"
    fl_pdb     = os.path.join("pdb", f"{pdb_prefix}.align.pdb")
    atom_list  = pr.atom.read(fl_pdb)
    atom_dict  = pr.atom.create_lookup_table(atom_list)

    print(f"Processing {pdb_prefix}")

    # Obtain coordinates...
    xyzs_dict = {}
    peptides = ["N", "CA", "C", "O"]
    for i in peptides:
        xyzs_dict[i] = pr.atom.extract_xyz_by_atom([i], atom_dict, chain, nterm_aux, cterm_aux)

    # Model with cubic spline...
    model_points = {}
    for i in peptides:
        model_points[i] = helix.tnb.fit_spline(xyzs_dict[i].T, num = num_turn * sample_per_rise)

    xyzs = xyzs_dict["CA"]

    # [[[ MEASURE CURVES IN SPACE ]]]
    xyzs_model = model_points["CA"]
    tvec, nvec, bvec, avec, \
    curv, tor, omega, r_list, p_list = helix.tnb.measure_spline(xyzs_model, offset = offterm * sample_per_rise)

    avec = avec.T
    avec = avec / np.linalg.norm(avec, axis = 1, keepdims = 1)

    # Calculate axial angle wrt the first axial vector...
    ## avec_mean = np.mean(avec, axis = 0)
    ## avec_mean /= np.linalg.norm(avec_mean)
    ## theta = avec @ avec_mean.reshape(-1, 1)
    avec_cum = np.cumsum(avec, axis = 0)
    avec_cum /= np.linalg.norm(avec_cum, axis = 1).reshape(-1,1)
    theta = avec_cum @ avec_cum[0].reshape(-1, 1)
    ## theta = avec @ avec[0].reshape(-1, 1)
    theta[theta > 1.0] = 1.0
    theta = np.arccos(theta) / np.pi * 180


    # [Visualize]
    drc = "helix"
    fl_eps = f"{pdb_prefix}.axialangle.eps"
    path_eps = os.path.join(drc, fl_eps)
    report(theta, r_list, p_list, omega, path_eps)
