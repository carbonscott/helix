#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "..")
## sys.path.insert(0, "/home/scott/Dropbox/codes/helix")

import pyrotein as pr
import numpy as np
import helix
import os
import GnuplotPy3

# Constants
drc_helix = "helix"
drc_pdb   = "pdb"
backbone  = ["N", "CA", "C", "O"]
atom_type = "CA"

# Read the reaction trajectory file...
# Trajectory is simulated by a shortest path solution subject to traveling
# salesman problem;  The scatter plot from SVD provides the "cities" (reaction
# state) to visit.  
fl_traj = "trajectory.dat"
traj_list = pr.utils.read_file(fl_traj)
traj_list = [ i[0] for i in traj_list ]

# Go through trajectories...
core_xyzs_dict      = {}
firstatom_xyzs_dict = {}
num_xyzs_dict       = {}
phase_dict          = {}
omega_dict          = {}
pitch_dict          = {}
radius_dict         = {}
term_dict           = {}
rmsd_dict           = {}
for traj_item in traj_list:
    # Read helixparam from files...
    pdb, chain = traj_item.split("_")
    entry      = f"{pdb}_{chain}"
    fl_param   = os.path.join(drc_helix, f"{entry}.align.helixparam.dat")
    lines      = pr.utils.read_file(fl_param)
    helixparam_dict = helix.parameterize.parse_helixparam_format(lines)

    # Select the segment...
    core_xyzs_dict[entry]      = {}
    firstatom_xyzs_dict[entry] = {}
    num_xyzs_dict[entry]       = {}
    phase_dict[entry]          = {}
    omega_dict[entry]          = {}
    pitch_dict[entry]          = {}
    radius_dict[entry]         = {}
    term_dict[entry]           = {}
    rmsd_dict[entry]           = {}

    # Get the modeled helical cores...
    for seg, v in helixparam_dict.items():
        # Fetch coordinates of a helix core...
        parvals     = v["param"]
        parval_dict = helix.parameterize.form_parval_dict(parvals)
        len_helix   = v["num"][atom_type]
        core_xyzs   = helix.parameterize.helixcore(parval_dict[atom_type], len_helix)
        core_xyzs_dict[entry][seg] = core_xyzs

        # Fetch coordinates of the first atom in the helix...
        firstatom_xyzs_dict[entry][seg] = v["firstatom"]

        # Fetch the number of atoms in the helix...
        num_xyzs_dict[entry][seg] = v["num"]

        # Fetch the phase of each helix...
        phase_dict[entry][seg] = v["phase"]

        # Fetch the radius of each helix...
        radius_dict[entry][seg] = v["radius"]

        # Recrod nterm and cterm...
        term_dict[entry][seg] = ( v["nterm"], v["cterm"] )

        # Record turn per residue...
        omega_dict[entry][seg] = v["omega"]

        # Record pitch...
        pitch_dict[entry][seg] = v["pitch"]

        # Record rmsd...
        rmsd_dict[entry][seg]  = v["rmsd"]




# Calculate distance between TM6e and TM7e
seg_ref, seg_tar = "TM6_e", "TM7_310"
atom_type = "CA"
inter_angle_dict = {}
inter_dist_dict  = {}
for traj_item in traj_list:
    pdb, chain = traj_item.split("_")
    entry      = f"{pdb}_{chain}"

    inter_angle_dict[entry] = helix.core.calc_interangle(core_xyzs_dict[entry][seg_ref], core_xyzs_dict[entry][seg_tar])
    inter_dist_dict[entry] = helix.core.calc_interdist(core_xyzs_dict[entry][seg_ref], core_xyzs_dict[entry][seg_tar])



# [[[ Visualize ]]]
len_pdb = len(traj_list)
fl_export = os.path.join(drc_helix, f"{seg_ref}-{seg_tar}.helixparam.eps")
gp = GnuplotPy3.GnuplotPy3()
gp("set encoding utf8")
gp("set terminal postscript eps  size 10, 4\\")
gp("                             enhanced color \\")
gp("                             font 'Helvetica,14' \\")
gp("                             linewidth 1")
gp(f"set output '{fl_export}'")
gp("set origin 0.0, 0.0")
gp("set lmargin 10.0")
gp("set multiplot layout 2, 1")

gp(f"set xrange [0:{len_pdb - 1}]")
gp(f"unset key")
gp(f"set grid xtics ytics linecolor '#666666'")

# Panel 1 - interhelical angle
gp("set bmargin 1.5")
gp("set ylabel 'Interhelical angle ({\260})'")
## gp("set yrange [70:120]")
for i, _ in enumerate(inter_angle_dict.keys()):
    gp(f"set xtics add (' ' {i}) rotate by 90 right font ',10'")

gp("plot \\")
gp(f"'-' using 1:2 with linespoints pointtype 7 pointsize 1.5 linecolor rgb 'blue',\\")
gp("")

for i, (_, v) in enumerate(inter_angle_dict.items()):
    vv = v
    vv *= 180 / np.pi
    gp(f"{i} {vv}")
gp("e")

# Panel 2 - interhelical distance
gp("set bmargin 4.0")
gp("set ylabel 'Interhelical distance ({\305})'")
## gp("set yrange [70:120]")
for i, _ in enumerate(inter_dist_dict.keys()):
    gp(f"set xtics add (' ' {i}) rotate by 90 right font ',10'")

for i, lbl in enumerate(inter_dist_dict.keys()):
    gp(f"set xtics add ('{lbl.replace('_','-')}' {i}) rotate by 90 right font ',10'")

gp("plot \\")
gp(f"'-' using 1:2 with linespoints pointtype 7 pointsize 1.5 linecolor rgb 'blue',\\")
gp("")

for i, (_, v) in enumerate(inter_dist_dict.items()):
    gp(f"{i} {v}")
gp("e")

gp("exit")
