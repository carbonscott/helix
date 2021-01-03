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
    nv = np.zeros(3)    # Axis vector
    for i in range(len(xyzs) - 3):
        h1 = bisector(xyzs[i], xyzs[i+1], xyzs[i+2])
        h2 = bisector(xyzs[i+1], xyzs[i+2], xyzs[i+3])
        hv = np.cross(h1, h2)
        nv += hv
    nv /= np.linalg.norm(nv)

    return nv


def helix(params, num, pt0):
    ''' Return modeled coordinates (x, y, z).
        The length of the helix is represented by the num of progress.  
        pt0 is the beginning position of the helix.
    '''
    # Unpack parameters...
    px    = params["px"]
    py    = params["py"]
    pz    = params["pz"]
    nx    = params["nx"]
    ny    = params["ny"]
    nz    = params["nz"]
    r     = params["r" ]
    s     = params["s"]
    omega = params["omega"]
    phi   = params["phi"]
    t     = params["t"]

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
    # Calculate residual...
    num   = xyzs.shape[0]
    ## input = px, py, pz, nx, ny, nz, r, s, omega, phi, t
    ## res   = helix(input, num, xyzs[0]) - xyzs
    res   = helix(params, num, xyzs[0]) - xyzs

    return res.reshape(-1)


def fit(params, xyzs, **kwargs):
    result = lmfit.minimize(residual, 
                            params, 
                            method = 'least_squares', 
                            args = (xyzs, ), 
                            **kwargs)
    return result


def protein(xyzs):
    ''' Fit a helix to amino acid according to input atomic positions.
    '''
    # Predefined helix parameters for amino acid...
    r             = 1.6
    s             = 5.5
    omega         = 1.7
    phi           = np.pi + omega
    t             = - 0.5 * np.linalg.norm(xyzs[-1] - xyzs[0])

    # Define init_values...
    # Assume helix axis passes through the average position of a straight helix
    px, py, pz = np.mean(xyzs, axis = 0)

    # Estimate the helix axis vector...
    nv = estimate_axis(xyzs)
    nx, ny, nz = nv

    # Init params...
    params = init_params()

    # Load init values
    params.add("px",    value = px)
    params.add("py",    value = py)
    params.add("pz",    value = pz)
    params.add("nx",    value = nx)
    params.add("ny",    value = ny)
    params.add("nz",    value = nz)
    params.add("r" ,    value = r)
    params.add("s" ,    value = s)
    params.add("omega", value = omega)
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


def check_fit(params, xyzs, pv0, nv0):
    # Generate the helix...
    q = helix(params, xyzs.shape[0], xyzs[0])

    # Unpack parameters...
    px    = params["px"]
    py    = params["py"]
    pz    = params["pz"]
    nx    = params["nx"]
    ny    = params["ny"]
    nz    = params["nz"]
    r     = params["r" ]
    s     = params["s"]
    omega = params["omega"]
    phi   = params["phi"]
    t     = params["t"]
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
        gp(f"{x} {y} {z} {i}")
    gp( "e")

    gp(f"{pv0[0]} {pv0[1]} {pv0[2]}")
    gp( "e")
    gp(f"{pv[0]} {pv[1]} {pv[2]}")
    gp( "e")

    for i, (x, y, z) in enumerate(q):
        gp(f"{x} {y} {z}")
    gp( "e")

    input("Press enter to exit...")


def check_select(params, xyzs, pv0, nv0, nterm, bindex, helixlen):
    # Generate the helix...
    xyzs_sel = xyzs[bindex:bindex+helixlen]
    q = helix(params, xyzs_sel.shape[0], xyzs_sel[0])

    # Unpack parameters...
    px    = params["px"]
    py    = params["py"]
    pz    = params["pz"]
    nx    = params["nx"]
    ny    = params["ny"]
    nz    = params["nz"]
    r     = params["r" ]
    s     = params["s"]
    omega = params["omega"]
    phi   = params["phi"]
    t     = params["t"]
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



def report_params(params, title = ""):
    # Unpack parameters...
    px    = params["px"].value
    py    = params["py"].value
    pz    = params["pz"].value
    nx    = params["nx"].value
    ny    = params["ny"].value
    nz    = params["nz"].value
    r     = params["r" ].value
    s     = params["s"].value
    omega = params["omega"].value
    phi   = params["phi"].value
    t     = params["t"].value

    print(f"{title}")
    print(f"params      value(s)")
    print(f"----------  --------")
    print(f"px, py, pz: {px:<10.3f}    {py:<10.3f}    {pz:<10.3f}")
    print(f"nx, ny, xz: {nx:<10.3f}    {ny:<10.3f}    {nz:<10.3f}")
    print(f"r:          {r:<10.3f}")
    print(f"s:          {s:<10.3f}")
    print(f"omega:      {omega:<10.3f}")
    print(f"phi:        {phi:<10.3f}")
    print(f"t:          {t:<10.3f}")
    print("")

    return None
