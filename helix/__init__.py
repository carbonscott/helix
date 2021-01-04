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


def helix(parvals, num, pt0):
    ''' Return modeled coordinates (x, y, z).
        The length of the helix is represented by the num of progress.  
        pt0 is the beginning position of the helix.
    '''
    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega, r, phi, t = parvals

    # Consider phase shift...
    psi_list = np.array([ omega * i for i in range(num) ])

    # Get vector connecting p and first atom...
    pv  = np.array([px, py, pz])
    pt0 = np.array(pt0)
    c = pt0 - pv

    # Form a orthonormal system...
    # Direction cosine: http://www.geom.uiuc.edu/docs/reference/CRC-formulas/node52.html
    n  = np.array([nx, ny, nz], dtype = np.float32)
    n /= np.linalg.norm(n)

    # Derive the v, w vector (third vector) to facilitate the construction of a helix...
    v = np.cross(n, c)
    w = np.cross(n, v)
    v /= np.linalg.norm(v)
    w /= np.linalg.norm(w)

    # Model it and save result in q...
    p  = np.array([px, py, pz], dtype = np.float32)
    q  = np.zeros((len(psi_list), 3))
    q += p.reshape(1, -1)
    q += n.reshape(1, -1) * s * psi_list.reshape(-1, 1) / (2 * np.pi)
    q += n.reshape(1, -1) * t
    q += v.reshape(1, -1) * r * np.cos(psi_list.reshape(-1, 1) + phi) + \
         w.reshape(1, -1) * r * np.sin(psi_list.reshape(-1, 1) + phi)

    return q


def residual(params, xyzs):
    # Calculate size of the helix...
    num   = xyzs.shape[0]

    # Avoid np.nan as the first valid point
    xyzs_nonan = xyzs[~np.isnan(xyzs).any(axis = 1)]

    # Unpack parameters...
    parvals = unpack_params(params)

    # Compute...
    res   = helix(parvals, num, xyzs_nonan[0]) - xyzs

    return res.reshape(-1)


def residual_peptide(params, xyzs_dict):
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

    # Create dictionary to store values
    num_dict        = {}
    xyzs_nonan_dict = {}
    res_dict        = {}

    # Computation for each type of atom...
    for i in xyzs_dict.keys():
        num_dict[i]  = xyzs_dict[i].shape[0]

        # Avoid np.nan as the first valid point
        xyzs_nonan_dict[i] = xyzs_dict[i][~np.isnan(xyzs_dict[i]).any(axis = 1)]

        # Compute...
        res = helix(parval_dict[i], num_dict[i], xyzs_nonan_dict[i][0]) - xyzs_dict[i]
        res_dict[i] = res

    # Format results for minimization...
    num_coords = np.sum( [ v for _, v in num_dict.items() ] )
    res_matrix = np.zeros( (num_coords, 3) )

    # Assign values...
    idx = 0
    for i, v in res_dict.items():
        res_matrix[idx:idx + num_dict[i], :] = v
        idx += num_dict[i]

    return res_matrix.reshape(-1)


def fit_peptide(params, xyzs_dict, **kwargs):
    result = lmfit.minimize(residual_peptide, 
                            params, 
                            method     = 'least_squares', 
                            nan_policy = 'omit',
                            args       = (xyzs_dict, ), 
                            **kwargs)
    return result


def fit(params, xyzs, **kwargs):
    result = lmfit.minimize(residual, 
                            params, 
                            method     = 'least_squares', 
                            nan_policy = 'omit',
                            args       = (xyzs, ), 
                            **kwargs)
    return result


def peptide(xyzs_dict):
    ''' Fit a helix to amino acid according to input atomic positions.
    '''
    # Predefined helix parameters for amino acid...
    s     = 5.5                  # pitch size of an alpha helix
    omega = 100 / 180 * np.pi    # 100 degree turn per residue

    # Predefine radius and phase offset
    r, phi = {}, {}
    r["N"] = 1.458    # Obtained from results of fitting N alone
    r["CA"] = 2.27
    r["C"] = 1.729
    r["O"] = 2.051
    phi["N"] = 4.85
    phi["CA"] = 4.98
    phi["C"] = 5.16
    phi["O"] = 5.152

    # Define init_values
    # Estimate the mean helix axis...
    nv_dict = {}
    for i in xyzs_dict.keys(): nv_dict[i] = estimate_axis(xyzs_dict[i])
    nv_array = np.array( [ v for v in nv_dict.values() ] )
    nv = np.nanmean(nv_array, axis = 0)
    nx, ny, nz = nv

    # Estimate the mean position that the axis of helix passes through...
    pv_dict = {}
    for i in xyzs_dict.keys(): pv_dict[i] = np.nanmean(xyzs_dict[i], axis = 0)
    pv_array = np.array( [ v for v in pv_dict.values() ] )
    pv = np.nanmean(pv_array, axis = 0)
    px, py, pz = pv

    # Predefine axial translational offset
    t = {}
    xyzs_nonan_dict = {}
    for i in xyzs_dict.keys():
        xyzs_nonan_dict[i] = xyzs_dict[i][~np.isnan(xyzs_dict[i]).any(axis = 1)]
        t[i] = - np.linalg.norm(pv - xyzs_nonan_dict[i][0])

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

    report_params_peptide(params, title = f"Init report")

    # Fitting process...
    # Set constraints...
    for i in params.keys(): params[i].set(vary = False)

    for i in ["px", "py", "pz"]: params[i].set(vary = True)
    result = fit_peptide(params, xyzs_dict)
    report_params_peptide(params, title = f"px, py, pz: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["phiN", "phiCA", "phiC", "phiO"]: params[i].set(vary = True)
    result = fit_peptide(params, xyzs_dict)
    report_params_peptide(params, title = f"phi: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["s", "omega"]: params[i].set(vary = True)
    result = fit_peptide(params, xyzs_dict)
    report_params_peptide(params, title = f"s, omega: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["tN", "tCA", "tC", "tO"]: params[i].set(vary = True)
    result = fit_peptide(params, xyzs_dict)
    report_params_peptide(params, title = f"t: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["rN", "rCA", "rC", "rO"]: params[i].set(vary = True)
    result = fit_peptide(params, xyzs_dict)
    report_params_peptide(params, title = f"r: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["nx", "ny", "nz"]: params[i].set(vary = True)
    result = fit_peptide(params, xyzs_dict)
    report_params_peptide(params, title = f"nx, ny, nz: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in range(5):
        result = fit_peptide(params, xyzs_dict, ftol = 1e-9)
        report_params_peptide(params, title = f"All params: " + \
                                      f"success = {result.success}, " + \
                                      f"cost = {result.cost}")
        params = result.params

    return result


def check_fit_peptide(params, xyzs_dict, pv0, nv0, nterm):
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

    for i in xyzs_dict.keys():
        print(f"Check {i}")
        check_fit(parval_dict[i], xyzs_dict[i], pv0, nv0, nterm)

    return None


def protein(xyzs):
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
    report_params(params, title = f"Init report")

    for i in ["px", "py", "pz"]: params[i].set(vary = True)
    result = fit(params, xyzs)
    report_params(params, title = f"px, py, pz: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["phi"]: params[i].set(vary = True)
    result = fit(params, xyzs)
    report_params(params, title = f"phi: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["s", "omega"]: params[i].set(vary = True)
    result = fit(params, xyzs)
    report_params(params, title = f"s, omega: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["t"]: params[i].set(vary = True)
    result = fit(params, xyzs)
    report_params(params, title = f"t: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["r"]: params[i].set(vary = True)
    result = fit(params, xyzs)
    report_params(params, title = f"r: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in ["nx", "ny", "nz"]: params[i].set(vary = True)
    result = fit(params, xyzs)
    report_params(params, title = f"nx, ny, nz: " + \
                                  f"success = {result.success}, " + \
                                  f"cost = {result.cost}")
    params = result.params

    for i in range(5):
        result = fit(params, xyzs, ftol = 1e-9)
        report_params(params, title = f"All params: " + \
                                      f"success = {result.success}, " + \
                                      f"cost = {result.cost}")
        params = result.params

    return result


def protein_fit_by_length(xyzs, helixlen):
    ''' Go through whole data and return the helix segment position that fits the best.
        The best fit is found using brute-force.
    '''
    assert len(xyzs) >= helixlen, "helixlen should be smaller than the total length."

    results = []
    for i in range(len(xyzs) - helixlen): 
        results.append( [ i, protein(xyzs[i:i+helixlen]) ] )

    sorted_results = sorted(results, key = lambda x: x[1].cost)

    return sorted_results[0]


def check_fit(parvals, xyzs, pv0, nv0, nterm):
    # Generate the helix...
    xyzs_nonan = xyzs[~np.isnan(xyzs).any(axis = 1)]
    ## parvals = unpack_params(params)
    q = helix(parvals, xyzs.shape[0], xyzs_nonan[0])

    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega, r, phi, t = parvals
    pv = np.array([px, py, pz])
    nv = np.array([nx, ny, nz])

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


def check_select(params, xyzs, pv0, nv0, nterm, bindex, helixlen):
    # Generate the helix...
    xyzs_sel = xyzs[bindex:bindex+helixlen]
    xyzs_sel_nonan = xyzs_sel[~np.isnan(xyzs_sel).any(axis = 1)]
    parvals = unpack_params(params)
    q = helix(parvals, xyzs_sel.shape[0], xyzs_sel_nonan[0])

    # Unpack parameters...
    px, py, pz, nx, ny, nz, s, omega, r, phi, t = parvals
    pv = np.array([px, py, pz])
    nv = np.array([nx, ny, nz])

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


def report_params_peptide(params, title = ""):
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
    print(f"omega:                   {omega:<10.3f}")
    print(f"rN, rCA, rC, rO:         {rN:<10.3f}    {rCA:<10.3f}    {rC:<10.3f}    {rO:<10.3f}")
    print(f"phiN, phiCA, phiC, phiO: {phiN:<10.3f}    {phiCA:<10.3f}    {phiC:<10.3f}    {phiO:<10.3f}")
    print(f"tN, tCA, tC, tO:         {tN:<10.3f}    {tCA:<10.3f}    {tC:<10.3f}    {tO:<10.3f}")
    print("")

    return None


def report_params(params, title = ""):
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
