#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 
# LMFIT for least square fitting
# https://lmfit.github.io/lmfit-py/intro.html
# pip install git+https://github.com/lmfit/lmfit-py.git --upgrade --user
#
import lmfit
import numpy as np


def init_params(): return lmfit.Parameters()
def unpack_params(params): return [ v.value  for _, v in params.items() ]


def bisector(xyz1, xyz2, xyz3):
    ''' Khan method.  
    '''
    v12 = xyz1 - xyz2
    v32 = xyz3 - xyz2
    return v12 + v32


def estimate_axis(xyzs):
    ''' Find an init axis.
        Refer to DOI: 10.1016/0097-8485(89)85005-3 for details.
    '''
    # Remove np.nan
    xyzs_nonan = xyzs[~np.isnan(xyzs).any(axis = 1)]

    # Compute
    nv = np.zeros(3)    # Axis vector
    for i in range(len(xyzs_nonan) - 3):
        h1 = bisector(xyzs_nonan[i], xyzs_nonan[i+1], xyzs_nonan[i+2])
        h2 = bisector(xyzs_nonan[i+1], xyzs_nonan[i+2], xyzs_nonan[i+3])
        hv = np.cross(h1, h2)
        nv += hv
    nv /= np.linalg.norm(nv)

    return nv


def helixmodel(parvals, num, pt0):
    ''' Return modeled coordinates (x, y, z).
        The length of the helix is represented by the num of progress.  
        pt0 is the beginning position of the helix.
    '''
    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega, r, phi, t = parvals

    # Consider phase shift...
    psi_list = np.array([ omega * i for i in range(num) ], dtype = np.float64)

    # Get vector connecting p and first atom...
    pv  = np.array([px, py, pz], dtype = np.float64)
    pt0 = np.array(pt0)
    c = pt0 - pv

    # Form a orthonormal system...
    # Direction cosine: http://www.geom.uiuc.edu/docs/reference/CRC-formulas/node52.html
    n  = np.array([nx, ny, nz], dtype = np.float64)
    n  = np.cos(n)

    # Derive the v, w vector (third vector) to facilitate the construction of a helix...
    v = np.cross(n, c)
    w = np.cross(n, v)
    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)

    # Model it and save result in q...
    p  = np.array([px, py, pz], dtype = np.float64)
    q  = np.zeros((len(psi_list), 3))
    q += p.reshape(1, -1)
    q += n.reshape(1, -1) * s * psi_list.reshape(-1, 1) / (2 * np.pi)
    q += n.reshape(1, -1) * t
    q += v.reshape(1, -1) * r * np.cos(psi_list.reshape(-1, 1) + phi) + \
         w.reshape(1, -1) * r * np.sin(psi_list.reshape(-1, 1) + phi)

    return q


def residual_purehelix(params, xyzs):
    # Calculate size of the helix...
    num   = xyzs.shape[0]

    # Avoid np.nan as the first valid point
    xyzs_nonan = xyzs[~np.isnan(xyzs).any(axis = 1)]

    # Unpack parameters...
    parvals = unpack_params(params)

    # Compute...
    res   = helixmodel(parvals, num, xyzs_nonan[0]) - xyzs

    return res.reshape(-1)


def residual_helix(params, xyzs_dict, pa0, lam):
    # Unpack parameters...
    parvals = unpack_params(params)
    px, py, pz, nx, ny, nz, s, omega = parvals[ :8]
    rN, rCA, rC, rO                  = parvals[8:8+4]
    phiN, phiCA, phiC, phiO          = parvals[12:12+4]
    tN, tCA, tC, tO                  = parvals[16:16+4]

    # Construct paramters for each atom...
    parval_dict = {}
    parval_dict["N"]  = px, py, pz, nx, ny, nz, s, omega, rN,  phiN,  tN
    parval_dict["CA"] = px, py, pz, nx, ny, nz, s, omega, rCA, phiCA, tCA
    parval_dict["C"]  = px, py, pz, nx, ny, nz, s, omega, rC,  phiC,  tC
    parval_dict["O"]  = px, py, pz, nx, ny, nz, s, omega, rO,  phiO,  tO

    # Consider residuals...
    # Create dictionary to store values
    num_dict        = {}
    xyzs_nonan_dict = {}
    res_dict        = {}

    # Computation for each type of atom
    for i in xyzs_dict.keys():
        num_dict[i]  = xyzs_dict[i].shape[0]

        # Avoid np.nan as the first valid point
        xyzs_nonan_dict[i] = xyzs_dict[i][~np.isnan(xyzs_dict[i]).any(axis = 1)]

        # Compute
        res = helixmodel(parval_dict[i], num_dict[i], xyzs_nonan_dict[i][0]) \
              - xyzs_dict[i]
        res_dict[i] = res

    # Format results for minimization
    num_coords = np.sum( [ v for _, v in num_dict.items() ] )
    res_matrix = np.zeros( (num_coords, 3) )

    # Assign values
    idx = 0
    for i, v in res_dict.items():
        res_matrix[idx:idx + num_dict[i], :] = v
        idx += num_dict[i]

    # Consider regularization (penalty)...
    # Facilitate regularization based on pa0
    pv  = np.array([px, py, pz], dtype = np.float64)
    pen_matrix = np.sqrt(lam[0]) * np.linalg.norm( pv - pa0 )

    # Combine residual and penalty...
    # [Warning!!!] It may lead to a performance hit
    comb_matrix = np.hstack( (res_matrix.reshape(-1), pen_matrix) )

    return comb_matrix.reshape(-1)


def fit_helix(params, xyzs_dict, pa0, lam, **kwargs):
    result = lmfit.minimize(residual_helix, 
                            params, 
                            method     = 'least_squares', 
                            nan_policy = 'omit',
                            args       = (xyzs_dict, pa0, lam), 
                            **kwargs)

    return result


def fit_purehelix(params, xyzs, **kwargs):
    result = lmfit.minimize(residual_purehelix, 
                            params, 
                            method     = 'least_squares', 
                            nan_policy = 'omit',
                            args       = (xyzs, ), 
                            **kwargs)
    return result


def purehelix(xyzs):
    ''' Fit a helix to amino acid according to input atomic positions.
    '''
    # Predefined helix parameters for amino acid...
    s             = 5.5
    omega         = 1.7
    r             = 1.6
    phi           = np.pi + omega

    # Define init_values...
    # Assume helix axis passes through the average position of a straight helix
    pv = np.nanmean(xyzs, axis = 0)
    px, py, pz = pv

    # Estimate the helix axis vector...
    nv = estimate_axis(xyzs)
    nx, ny, nz = nv

    # Avoid nan when estimating t
    xyzs_nonan = xyzs[~np.isnan(xyzs).any(axis = 1)]
    t = - np.linalg.norm(pv - xyzs_nonan[0])

    # Init params...
    params = init_params()

    # Load init values
    params.add("px",    value = px)
    params.add("py",    value = py)
    params.add("pz",    value = pz)
    params.add("nx",    value = nx)
    params.add("ny",    value = ny)
    params.add("nz",    value = nz)
    params.add("s" ,    value = s)
    params.add("omega", value = omega)
    params.add("r" ,    value = r)
    params.add("phi",   value = phi)
    params.add("t",     value = t)

    # Set constraints...
    for i in params.keys(): params[i].set(vary = False)

    # Fitting process...
    report_params_purehelix(params, title = f"Init report")

    for i in ["px", "py", "pz"]: params[i].set(vary = True)
    result = fit_purehelix(params, xyzs)
    report_params_purehelix(params, title = f"px, py, pz: " + \
                                  f"success = {result.success}, " + \
                                  f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["phi"]: params[i].set(vary = True)
    result = fit_purehelix(params, xyzs)
    report_params_purehelix(params, title = f"phi: " + \
                                  f"success = {result.success}, " + \
                                  f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["s", "omega"]: params[i].set(vary = True)
    result = fit_purehelix(params, xyzs)
    report_params_purehelix(params, title = f"s, omega: " + \
                                  f"success = {result.success}, " + \
                                  f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["t"]: params[i].set(vary = True)
    result = fit_purehelix(params, xyzs)
    report_params_purehelix(params, title = f"t: " + \
                                  f"success = {result.success}, " + \
                                  f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["r"]: params[i].set(vary = True)
    result = fit_purehelix(params, xyzs)
    report_params_purehelix(params, title = f"r: " + \
                                  f"success = {result.success}, " + \
                                  f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["nx", "ny", "nz"]: params[i].set(vary = True)
    result = fit_purehelix(params, xyzs)
    report_params_purehelix(params, title = f"nx, ny, nz: " + \
                                  f"success = {result.success}, " + \
                                  f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in range(5):
        result = fit_purehelix(params, xyzs, ftol = 1e-9)
        report_params_purehelix(params, title = f"All params: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
        params = result.params

    return result


def helix(xyzs_dict, lam, report = True):
    ''' Fit a helix to amino acid according to input atomic positions.
        lam is regularization constant.  
    '''
    # Predefined helix parameters for amino acid...
    s     = 5.5                  # pitch size of an alpha helix
    omega = 100 / 180 * np.pi    # 100 degree turn per residue

    # Predefine radius and phase offset
    r, phi    = {}, {}
    r["N"]    = 1.458    # Obtained from results of fitting N alone
    r["CA"]   = 2.27
    r["C"]    = 1.729
    r["O"]    = 2.051
    phi["N"]  = - np.pi / 2
    phi["CA"] = - np.pi / 2
    phi["C"]  = - np.pi / 2
    phi["O"]  = - np.pi / 2

    # Define init_values
    # Estimate the mean helix axis...
    nv_dict = {}
    for i in xyzs_dict.keys(): nv_dict[i] = estimate_axis(xyzs_dict[i])
    nv_array = np.array( [ v for v in nv_dict.values() ] )
    nv = np.nanmean(nv_array, axis = 0)
    nv /= np.linalg.norm(nv)

    # Calculate direction cosine angle
    # nx = cos(alpha) = a1/|a|
    # ny = cos(beta ) = a2/|a|
    # nz = cos(gamma) = a3/|a|
    nx, ny, nz = np.arccos(nv)

    # Estimate the mean position that the axis of helix passes through...
    pv_dict = {}
    for i in xyzs_dict.keys(): pv_dict[i] = np.nanmean(xyzs_dict[i], axis = 0)
    pv_array = np.array( [ v for v in pv_dict.values() ], dtype = np.float64)
    pv = np.nanmean(pv_array, axis = 0)
    px, py, pz = pv

    # Record the init position that an axial will pass through
    pa0 = pv.copy()

    # Predefine axial translational offset
    t = {}
    xyzs_nonan_dict = {}
    for i in xyzs_dict.keys():
        xyzs_nonan_dict[i] = xyzs_dict[i][~np.isnan(xyzs_dict[i]).any(axis = 1)]
        ## t[i] = - np.linalg.norm(pv - xyzs_nonan_dict[i][0])
        # The projection along axis is equivalent to translation...
        pv_to_firstatom = xyzs_nonan_dict[i][0] - pv
        t[i] = np.dot( pv_to_firstatom, nv )
    ## t = {}
    ## t["N"] = -8.053
    ## t["CA"] = -6.808
    ## t["C"] = -5.338
    ## t["O"] = -3.691

    # Init params...
    params = init_params()

    # Load init values
    params.add("px"   , value = px)
    params.add("py"   , value = py)
    params.add("pz"   , value = pz)
    params.add("nx"   , value = nx)
    params.add("ny"   , value = ny)
    params.add("nz"   , value = nz)
    params.add("s"    , value = s)
    params.add("omega", value = omega)
    params.add("rN"   , value = r["N"])
    params.add("rCA"  , value = r["CA"])
    params.add("rC"   , value = r["C"])
    params.add("rO"   , value = r["C"])
    params.add("phiN" , value = phi["N"])
    params.add("phiCA", value = phi["CA"])
    params.add("phiC" , value = phi["C"])
    params.add("phiO" , value = phi["O"])
    params.add("tN"   , value = t["N"])
    params.add("tCA"  , value = t["CA"])
    params.add("tC"   , value = t["C"])
    params.add("tO"   , value = t["O"])

    if report: report_params_helix(params, title = f"Init report")

    # Fitting process...
    # Set constraints...
    for i in params.keys(): params[i].set(vary = False)

    for i in ["px", "py", "pz"]: params[i].set(vary = True)
    result = fit_helix(params, xyzs_dict, pa0, lam)
    if report:
        report_params_helix(params, title = f"px, py, pz: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["nx", "ny", "nz"]: params[i].set(vary = True)
    result = fit_helix(params, xyzs_dict, pa0, lam)
    if report:
        report_params_helix(params, title = f"nx, ny, nz: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["phiN", "phiCA", "phiC", "phiO"]: params[i].set(vary = True)
    result = fit_helix(params, xyzs_dict, pa0, lam)
    if report:
        report_params_helix(params, title = f"phi: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["s", "omega"]: params[i].set(vary = True)
    result = fit_helix(params, xyzs_dict, pa0, lam)
    if report:
        report_params_helix(params, title = f"s, omega: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["tN", "tCA", "tC", "tO"]: params[i].set(vary = True)
    result = fit_helix(params, xyzs_dict, pa0, lam)
    if report:
        report_params_helix(params, title = f"t: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in ["rN", "rCA", "rC", "rO"]: params[i].set(vary = True)
    result = fit_helix(params, xyzs_dict, pa0, lam)
    if report:
        report_params_helix(params, title = f"r: " + \
                                      f"success = {result.success}, " + \
                                      f"rmsd = {calc_rmsd(result.residual)}")
    params = result.params

    for i in range(5):
        result = fit_helix(params, xyzs_dict, pa0, lam, ftol = 1e-9)
        if report:
            report_params_helix(params, title = f"All params: " + \
                                          f"success = {result.success}, " + \
                                          f"rmsd = {calc_rmsd(result.residual)}")
        params = result.params

    return result


def check_fit_helix(params, xyzs_dict, pv0, nv0, nterm, atom_to_check):
    # Unpack parameters...
    parvals = unpack_params(params)
    px, py, pz, nx, ny, nz, s, omega = parvals[ :8]
    rN, rCA, rC, rO                  = parvals[8:8+4]
    phiN, phiCA, phiC, phiO          = parvals[12:12+4]
    tN, tCA, tC, tO                  = parvals[16:16+4]

    # Construct paramters for each atom...
    parval_dict = {}
    parval_dict["N"]  = px, py, pz, nx, ny, nz, s, omega, rN,  phiN,  tN
    parval_dict["CA"] = px, py, pz, nx, ny, nz, s, omega, rCA, phiCA, tCA
    parval_dict["C"]  = px, py, pz, nx, ny, nz, s, omega, rC,  phiC,  tC
    parval_dict["O"]  = px, py, pz, nx, ny, nz, s, omega, rO,  phiO,  tO

    for i in atom_to_check:
        print(f"Check {i}")
        check_fit_purehelix(parval_dict[i], xyzs_dict[i], pv0, nv0, nterm)

    return None


def fit_helix_by_length(xyzs_dict, helixlen):
    ''' Go through whole data and return the helix segment position that fits the best.
        The best fit is found using brute-force.
    '''
    assert len(xyzs_dict["N"]) >= helixlen, "helixlen should be smaller than the total length."

    results = []
    xyzs_filtered_dict = {}
    for i in range(len(xyzs_dict["N"]) - helixlen): 
        for k, v in xyzs_dict.items():
            xyzs_filtered_dict[k] = v[i:i+helixlen]
        results.append( [ i, helix(xyzs_filtered_dict) ] )

    sorted_results = sorted(results, key = lambda x: calc_rmsd(x[1].residual))

    return sorted_results[0]


def fit_purehelix_by_length(xyzs, helixlen):
    ''' Go through whole data and return the helix segment position that fits the best.
        The best fit is found using brute-force.
    '''
    assert len(xyzs) >= helixlen, "helixlen should be smaller than the total length."

    results = []
    for i in range(len(xyzs) - helixlen): 
        results.append( [ i, purehelix(xyzs[i:i+helixlen]) ] )

    sorted_results = sorted(results, key = lambda x: calc_rmsd(x[1].residual))

    return sorted_results[0]


def check_fit_purehelix(parvals, xyzs, pv0, nv0, nterm):
    # Generate the helix...
    xyzs_nonan = xyzs[~np.isnan(xyzs).any(axis = 1)]
    q = helixmodel(parvals, xyzs.shape[0], xyzs_nonan[0])

    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega, r, phi, t = parvals
    pv = np.array([px, py, pz])
    nv = np.array([nx, ny, nz])
    nv = np.cos(nv)

    import GnuplotPy3
    gp = GnuplotPy3.GnuplotPy3()

    gp("set view equal xyz")
    gp("set xlabel 'x'")
    gp("set ylabel 'y'")
    gp("set key")
    gp(f"set arrow front from {pv0[0]},{pv0[1]},{pv0[2]}  \
                           to {pv0[0] + nv0[0]}, \
                              {pv0[1] + nv0[1]}, \
                              {pv0[2] + nv0[2]}  \
                         linecolor rgb 'black'")
    gp(f"set arrow front from {pv[0]},{pv[1]},{pv[2]}  \
                           to {pv[0] + nv[0]}, \
                              {pv[1] + nv[1]}, \
                              {pv[2] + nv[2]}  \
                         linecolor rgb 'red'")
    gp(f"splot '-' using 1:2:3   with linespoints pointtype 6 linecolor rgb 'black' title 'data', \\")
    gp(f"      '-' using 1:2:3:4 with labels notitle, \\")
    gp(f"      '-' using 1:2:3   with points pointtype 6 linecolor rgb 'black'notitle, \\")
    gp(f"      '-' using 1:2:3   with points pointtype 6 linecolor rgb 'red'notitle, \\")
    gp(f"      '-' using 1:2:3   with linespoints pointtype 6 linecolor rgb 'red' title 'model', \\")
    gp(f"")

    for i, (x, y, z) in enumerate(xyzs):
        if np.nan in (x, y, z): continue
        gp(f"{x} {y} {z}")
    gp( "e")
    for i, (x, y, z) in enumerate(xyzs):
        if np.nan in (x, y, z): continue
        gp(f"{x} {y} {z} {i + nterm}")
    gp( "e")

    gp(f"{pv0[0]} {pv0[1]} {pv0[2]}")
    gp( "e")
    gp(f"{pv[0]} {pv[1]} {pv[2]}")
    gp( "e")

    for i, (x, y, z) in enumerate(q):
        if np.nan in (x, y, z): continue
        gp(f"{x} {y} {z}")
    gp( "e")

    input("Press enter to exit...")

    return None


def check_select_helix(parvals, xyzs_dict, pv0, nv0, nterm, bindex, helixlen):
    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega = parvals[ :8]
    rN, rCA, rC, rO                  = parvals[8:8+4]
    phiN, phiCA, phiC, phiO          = parvals[12:12+4]
    tN, tCA, tC, tO                  = parvals[16:16+4]

    # Construct paramters for each atom...
    parval_dict = {}
    parval_dict["N"]  = px, py, pz, nx, ny, nz, s, omega, rN,  phiN,  tN
    parval_dict["CA"] = px, py, pz, nx, ny, nz, s, omega, rCA, phiCA, tCA
    parval_dict["C"]  = px, py, pz, nx, ny, nz, s, omega, rC,  phiC,  tC
    parval_dict["O"]  = px, py, pz, nx, ny, nz, s, omega, rO,  phiO,  tO

    for i in xyzs_dict.keys():
        print(f"Check {i}")
        check_select_purehelix(parval_dict[i], xyzs_dict[i], pv0, nv0, nterm, bindex, helixlen)

    return None


def check_select_purehelix(parvals, xyzs, pv0, nv0, nterm, bindex, helixlen):
    # Generate the helix...
    xyzs_sel = xyzs[bindex:bindex+helixlen]
    xyzs_sel_nonan = xyzs_sel[~np.isnan(xyzs_sel).any(axis = 1)]
    ## parvals = unpack_params(params)
    q = helixmodel(parvals, xyzs_sel.shape[0], xyzs_sel_nonan[0])

    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega, r, phi, t = parvals
    pv = np.array([px, py, pz])
    nv = np.array([nx, ny, nz])
    nv = np.cos(nv)

    import GnuplotPy3
    gp = GnuplotPy3.GnuplotPy3()

    gp("set view equal xyz")
    gp("set xlabel 'x'")
    gp("set ylabel 'y'")
    gp("unset key")
    gp(f"set arrow front from {pv0[0]},{pv0[1]},{pv0[2]}  \
                           to {pv0[0] + nv0[0]}, \
                              {pv0[1] + nv0[1]}, \
                              {pv0[2] + nv0[2]}  \
                         linecolor rgb 'black'")
    gp(f"set arrow front from {pv[0]},{pv[1]},{pv[2]}  \
                           to {pv[0] + nv[0]}, \
                              {pv[1] + nv[1]}, \
                              {pv[2] + nv[2]}  \
                         linecolor rgb 'red'")
    gp(f"splot '-' using 1:2:3   with linespoints pointtype 6, \\")
    gp(f"      '-' using 1:2:3:4 with labels , \\")
    gp(f"      '-' using 1:2:3   with points pointtype 6 linecolor rgb 'black', \\")
    gp(f"      '-' using 1:2:3   with points pointtype 6 linecolor rgb 'red', \\")
    gp(f"      '-' using 1:2:3   with linespoints pointtype 6 linecolor rgb 'green', \\")
    gp(f"")

    for i, (x, y, z) in enumerate(xyzs):
        gp(f"{x} {y} {z}")
    gp( "e")
    for i, (x, y, z) in enumerate(xyzs):
        gp(f"{x} {y} {z} {i + nterm}")
    gp( "e")

    gp(f"{pv0[0]} {pv0[1]} {pv0[2]}")
    gp( "e")
    gp(f"{pv[0]} {pv[1]} {pv[2]}")
    gp( "e")

    for i, (x, y, z) in enumerate(q):
        gp(f"{x} {y} {z}")
    gp( "e")

    input("Press enter to exit...")

    return None


def report_params_helix(params, title = ""):
    # Unpack parameters...
    px    = params["px"   ].value
    py    = params["py"   ].value
    pz    = params["pz"   ].value
    nx    = params["nx"   ].value
    ny    = params["ny"   ].value
    nz    = params["nz"   ].value
    s     = params["s"    ].value
    omega = params["omega"].value
    rN    = params["rN"   ].value
    rCA   = params["rCA"  ].value
    rC    = params["rC"   ].value
    rO    = params["rO"   ].value
    phiN  = params["phiN" ].value
    phiCA = params["phiCA"].value
    phiC  = params["phiC" ].value
    phiO  = params["phiO" ].value
    tN    = params["tN"   ].value
    tCA   = params["tCA"  ].value
    tC    = params["tC"   ].value
    tO    = params["tO"   ].value

    print(f"{title}")
    print(f"params                   value(s)")
    print(f"----------               --------")
    print(f"px, py, pz:              {px:<10.3f}    {py:<10.3f}    {pz:<10.3f}")
    print(f"nx, ny, xz:              {nx:<10.3f}    {ny:<10.3f}    {nz:<10.3f}")
    print(f"s:                       {s:<10.3f}")
    print(f"omega:                   {omega / np.pi * 180:<10.3f}")
    print(f"rN, rCA, rC, rO:         {rN:<10.3f}    {rCA:<10.3f}    {rC:<10.3f}    {rO:<10.3f}")
    print(f"phiN, phiCA, phiC, phiO: {phiN / np.pi * 180:<10.3f}    {phiCA / np.pi * 180:<10.3f}    {phiC / np.pi * 180:<10.3f}    {phiO / np.pi * 180:<10.3f}")
    print(f"tN, tCA, tC, tO:         {tN:<10.3f}    {tCA:<10.3f}    {tC:<10.3f}    {tO:<10.3f}")
    print("")

    return None


def report_params_purehelix(params, title = ""):
    # Unpack parameters...
    px    = params["px"].value
    py    = params["py"].value
    pz    = params["pz"].value
    nx    = params["nx"].value
    ny    = params["ny"].value
    nz    = params["nz"].value
    s     = params["s"].value
    omega = params["omega"].value
    r     = params["r" ].value
    phi   = params["phi"].value
    t     = params["t"].value

    print(f"{title}")
    print(f"params      value(s)")
    print(f"----------  --------")
    print(f"px, py, pz: {px:<10.3f}    {py:<10.3f}    {pz:<10.3f}")
    print(f"nx, ny, xz: {nx:<10.3f}    {ny:<10.3f}    {nz:<10.3f}")
    print(f"s:          {s:<10.3f}")
    print(f"omega:      {omega:<10.3f}")
    print(f"r:          {r:<10.3f}")
    print(f"phi:        {phi:<10.3f}")
    print(f"t:          {t:<10.3f}")
    print("")

    return None


def calc_rmsd(residual):
    ''' https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    residual_nonan = residual[~np.isnan(residual)]
    res = np.nan
    if len(residual_nonan) > 0:
        res = np.sqrt( np.sum(residual_nonan * residual_nonan) / len(residual_nonan) )
    return res


def whole_helix(xyzs_dict, len_segment, step, nterm, cterm):
    ''' Go through the whole helix by step with a helix segment length len_segment.
    '''
    len_peptide = cterm - nterm + 1
    params_dict = {}
    for bindex in range(0, len_peptide - len_segment, step):
        # Obtain the helix segment...
        xyzs_filtered_dict = {}
        for k, v in xyzs_dict.items():
            xyzs_filtered_dict[k] = v[bindex:bindex+len_segment]

        # Fitting...
        try: result = helix(xyzs_filtered_dict)
        except ValueError: continue

        print(f"Fitting {bindex + nterm}...{bindex + nterm + len_segment}")

        # Save values...
        params_dict[bindex] = [result.params, calc_rmsd(result.residual)]

    return params_dict


def export_params_dict(params_dict, fl_out):
    with open(fl_out,'w') as fh:
        for k, v in params_dict.items():
            # Unpack values...
            params, rmsd = v

            # Unpack parameters...
            parvals = unpack_params(params)

            # Export...
            fh.write(f"{k:03d}")
            fh.write(f"{rmsd:10.5f}")
            for parval in parvals:
                fh.write(f"{parval:8.3f}")
                fh.write(f"    ")
            fh.write("\n")


def report_result(result):
    # Fetch params and rmsd...
    params = result.params
    rmsd   = calc_rmsd(result.residual)

    # Unpack params...
    parvals = unpack_params(params)
    px, py, pz, nx, ny, nz, s, omega = parvals[ :8]
    rN, rCA, rC, rO                  = parvals[8:8+4]
    phiN, phiCA, phiC, phiO          = parvals[12:12+4]
    tN, tCA, tC, tO                  = parvals[16:16+4]

    # Convert angle from radian to degree
    omega = omega / np.pi * 180
    phiN  = phiN  / np.pi * 180
    phiCA = phiCA / np.pi * 180
    phiC  = phiC  / np.pi * 180
    phiO  = phiO  / np.pi * 180

    # Export values...
    res = [ f"{i:10.3f}" \
            for i in [ px, py, pz, nx, ny, nz, s, omega,\
                       rN, rCA, rC, rO,                 \
                       phiN, phiCA, phiC, phiO,         \
                       tN, tCA, tC, tO,                 \
                       rmsd ] ]

    return res
