#!/usr/bin/env python3
"""
High-energy neutrino DIS cross-section benchmark
=================================================

Purpose
-------
Make an independent Standard-Model DIS cross-section curve that can be
plotted against your GENIE and Geant4 results.

This script calculates a leading-order (LO) parton-model DIS prediction
using the CT18NNLO PDF set through LHAPDF, including the W/Z propagators.

IMPORTANT:
    This is an analytic/theory benchmark, not the full NNLO calculation
    of Weigel, Conrad & Garcia-Soto (WCG24, arXiv:2408.05866).
    For a paper-quality three-way validation, use this script as a
    cross-check and, ideally, overlay the published WCG24 calculation
    or its tabulated cross-section data as a separate theory curve/band.

Default target:
    isoscalar nucleon, (p + n) / 2

Default energies:
    0.1, 0.3, 0.5, 1, 2, 3, 5 TeV

Outputs:
    theory_DIS_CT18NNLO_LO.csv
    theory_DIS_CT18NNLO_LO.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

try:
    import lhapdf
except ImportError:
    raise SystemExit(
        "\nLHAPDF is required.\n"
        "Install LHAPDF + the CT18NNLO PDF set, then run again.\n"
        "See the instructions below the script.\n"
    )

# ============================================================
# USER SETTINGS
# ============================================================

ENERGIES_GEV = np.logspace(np.log10(100.0), np.log10(5000.0), 100)

# Your initial production grid:
TEST_ENERGIES_GEV = np.array([100, 300, 500, 1000, 2000, 3000, 5000])

PDF_NAME = "CT18NNLO"
PDF_MEMBER = 0

# Integration settings
N_X = 80
N_Y = 80

# DIS cuts.
# These prevent the calculation from being dominated by regions where
# the simple massless LO DIS approximation is not appropriate.
Q2_MIN_GEV2 = 1.0
W2_MIN_GEV2 = 4.0

# Electroweak constants
GF = 1.1663787e-5       # GeV^-2
M_W = 80.379            # GeV
M_Z = 91.1876            # GeV
SIN2_THETA_W = 0.23122

# Nucleon mass
M_N = 0.938918           # GeV

# Conversion: 1 GeV^-2 = 0.389379338e-27 cm^2
GEV2_TO_CM2 = 0.389379338e-27

# ============================================================
# PDF
# ============================================================

pdf = lhapdf.mkPDF(PDF_NAME, PDF_MEMBER)

# ============================================================
# PDF HELPERS
# ============================================================

def xfx(pid, x, Q2):
    """
    Return x*f_i(x,Q2) from LHAPDF.

    pid:
       2 = u
      -2 = ubar
       1 = d
      -1 = dbar
       3 = s
      -3 = sbar
       4 = c
      -4 = cbar
       5 = b
      -5 = bbar
    """
    # LHAPDF uses Q, not Q^2
    Q = np.sqrt(Q2)

    # Stay inside the PDF validity range.
    # CT18NNLO's nominal x/Q range is much wider than needed here,
    # but clipping avoids numerical failures near integration boundaries.
    x_safe = np.clip(x, 1e-9, 1.0 - 1e-12)
    Q_safe = np.clip(Q, 1.3, 1.0e5)

    return np.asarray(pdf.xfxQ2(pid, x_safe, Q_safe**2))


def xf_isoscalar(pid, x, Q2):
    """
    Isoscalar PDF:
        f_iso = (f_proton + f_neutron)/2

    For u,d:
        u_n = d_p
        d_n = u_p
        ubar_n = dbar_p
        dbar_n = ubar_p

    Strange/charm/bottom are taken to be the same for p and n.
    """

    if pid == 2:      # u
        return 0.5 * (xfx(2, x, Q2) + xfx(1, x, Q2))
    if pid == 1:      # d
        return 0.5 * (xfx(1, x, Q2) + xfx(2, x, Q2))
    if pid == -2:     # ubar
        return 0.5 * (xfx(-2, x, Q2) + xfx(-1, x, Q2))
    if pid == -1:     # dbar
        return 0.5 * (xfx(-1, x, Q2) + xfx(-2, x, Q2))

    return xfx(pid, x, Q2)


# ============================================================
# GAUSS-LEGENDRE QUADRATURE
# ============================================================

nodes_x, weights_x = roots_legendre(N_X)
nodes_y, weights_y = roots_legendre(N_Y)

# y is integrated directly over [0,1].
y_values = 0.5 * (nodes_y + 1.0)
y_weights = 0.5 * weights_y

# x is integrated logarithmically:
# x = 10^u
# dx = ln(10) * 10^u du
U_MIN = -8.0
U_MAX = 0.0

u_values = 0.5 * (nodes_x + 1.0) * (U_MAX - U_MIN) + U_MIN
u_weights = 0.5 * weights_x * (U_MAX - U_MIN)

x_values = 10.0**u_values
dx_du = np.log(10.0) * x_values

# ============================================================
# CC DIFFERENTIAL CROSS SECTION
# ============================================================

def cc_integrand(E, x, y, neutrino=True):
    """
    LO neutrino-nucleon CC DIS:

    d2sigma/dxdy =
        (2 GF^2 M E / pi)
        * [MW^2/(Q2+MW^2)]^2
        * [quark + antiquark*(1-y)^2]

    For nu:
        d + s + b + (ubar + cbar)*(1-y)^2

    For antinu:
        (dbar + sbar + bbar) + (u + c)*(1-y)^2

    CKM effects are not explicitly included here.
    At this stage this is intended as a clean benchmark, not a
    precision NNLO prediction.
    """

    Q2 = 2.0 * M_N * E * x * y
    W2 = M_N**2 + Q2 * (1.0 / x - 1.0)

    if Q2 < Q2_MIN_GEV2 or W2 < W2_MIN_GEV2:
        return 0.0

    propagator = (M_W**2 / (Q2 + M_W**2))**2

    if neutrino:
        q = (
            xf_isoscalar(1, x, Q2) +     # d
            xf_isoscalar(3, x, Q2) +     # s
            xf_isoscalar(5, x, Q2)       # b
        )

        qbar = (
            xf_isoscalar(-2, x, Q2) +    # ubar
            xf_isoscalar(-4, x, Q2)      # cbar
        )

    else:
        q = (
            xf_isoscalar(-1, x, Q2) +    # dbar
            xf_isoscalar(-3, x, Q2) +    # sbar
            xf_isoscalar(-5, x, Q2)      # bbar
        )

        qbar = (
            xf_isoscalar(2, x, Q2) +     # u
            xf_isoscalar(4, x, Q2)       # c
        )

    prefactor = 2.0 * GF**2 * M_N * E / np.pi

    return prefactor * propagator * (q + qbar * (1.0 - y)**2)


# ============================================================
# NC DIFFERENTIAL CROSS SECTION
# ============================================================

def nc_integrand(E, x, y, neutrino=True):
    """
    LO neutrino-nucleon NC DIS using electroweak Z couplings.

    gL = T3 - Q sin^2(theta_W)
    gR = -Q sin^2(theta_W)

    For neutrinos:
       q:     gL^2 + gR^2 (1-y)^2
       qbar:  gR^2 + gL^2 (1-y)^2

    For antineutrinos the q/qbar weights are interchanged.
    """

    Q2 = 2.0 * M_N * E * x * y
    W2 = M_N**2 + Q2 * (1.0 / x - 1.0)

    if Q2 < Q2_MIN_GEV2 or W2 < W2_MIN_GEV2:
        return 0.0

    propagator = (M_Z**2 / (Q2 + M_Z**2))**2

    s2w = SIN2_THETA_W

    # Up-type quarks: u,c
    gL_u = 0.5 - (2.0 / 3.0) * s2w
    gR_u = -(2.0 / 3.0) * s2w

    # Down-type quarks: d,s,b
    gL_d = -0.5 + (1.0 / 3.0) * s2w
    gR_d = +(1.0 / 3.0) * s2w

    u = xf_isoscalar(2, x, Q2)
    ubar = xf_isoscalar(-2, x, Q2)
    c = xf_isoscalar(4, x, Q2)
    cbar = xf_isoscalar(-4, x, Q2)

    d = xf_isoscalar(1, x, Q2)
    dbar = xf_isoscalar(-1, x, Q2)
    s = xf_isoscalar(3, x, Q2)
    sbar = xf_isoscalar(-3, x, Q2)
    b = xf_isoscalar(5, x, Q2)
    bbar = xf_isoscalar(-5, x, Q2)

    up_q = u + c
    up_qbar = ubar + cbar

    down_q = d + s + b
    down_qbar = dbar + sbar + bbar

    if neutrino:
        integrand = (
            up_q * (gL_u**2 + gR_u**2 * (1.0-y)**2)
            + up_qbar * (gR_u**2 + gL_u**2 * (1.0-y)**2)
            + down_q * (gL_d**2 + gR_d**2 * (1.0-y)**2)
            + down_qbar * (gR_d**2 + gL_d**2 * (1.0-y)**2)
        )
    else:
        integrand = (
            up_q * (gR_u**2 + gL_u**2 * (1.0-y)**2)
            + up_qbar * (gL_u**2 + gR_u**2 * (1.0-y)**2)
            + down_q * (gR_d**2 + gL_d**2 * (1.0-y)**2)
            + down_qbar * (gL_d**2 + gR_d**2 * (1.0-y)**2)
        )

    prefactor = 2.0 * GF**2 * M_N * E / np.pi

    return prefactor * propagator * integrand


# ============================================================
# NUMERICAL INTEGRATION
# ============================================================

def integrate_cross_section(E, process="CC", neutrino=True):
    """
    Integrate over x and y using Gauss-Legendre quadrature.
    """

    total = 0.0

    for x, wx, jac_x in zip(x_values, u_weights, dx_du):

        subtotal = 0.0

        for y, wy in zip(y_values, y_weights):

            if process.upper() == "CC":
                value = cc_integrand(E, x, y, neutrino)
            elif process.upper() == "NC":
                value = nc_integrand(E, x, y, neutrino)
            else:
                raise ValueError("process must be 'CC' or 'NC'")

            subtotal += wy * value

        total += wx * jac_x * subtotal

    return total * GEV2_TO_CM2


# ============================================================
# CALCULATE CURVES
# ============================================================

def calculate_all(energies):
    results = {
        "E_GeV": energies,
        "nu_CC": [],
        "nu_NC": [],
        "nubar_CC": [],
        "nubar_NC": [],
    }

    for i, E in enumerate(energies):

        print(f"{i+1:3d}/{len(energies)}  E = {E:9.3f} GeV")

        results["nu_CC"].append(
            integrate_cross_section(E, "CC", True)
        )

        results["nu_NC"].append(
            integrate_cross_section(E, "NC", True)
        )

        results["nubar_CC"].append(
            integrate_cross_section(E, "CC", False)
        )

        results["nubar_NC"].append(
            integrate_cross_section(E, "NC", False)
        )

    for key in results:
        results[key] = np.asarray(results[key])

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results, filename="theory_DIS_CT18NNLO_LO.csv"):

    data = np.column_stack([
        results["E_GeV"],
        results["nu_CC"],
        results["nu_NC"],
        results["nubar_CC"],
        results["nubar_NC"],
    ])

    header = (
        "E_GeV,"
        "nu_CC_cm2_per_nucleon,"
        "nu_NC_cm2_per_nucleon,"
        "nubar_CC_cm2_per_nucleon,"
        "nubar_NC_cm2_per_nucleon"
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header,
        comments=""
    )

    print(f"\nSaved: {filename}")


# ============================================================
# PLOT
# ============================================================

def make_plot(results, filename="theory_DIS_CT18NNLO_LO.png"):

    E_TeV = results["E_GeV"] / 1000.0

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    ax.loglog(
        E_TeV,
        results["nu_CC"],
        label=r"$\nu$ CC"
    )

    ax.loglog(
        E_TeV,
        results["nu_NC"],
        linestyle="--",
        label=r"$\nu$ NC"
    )

    ax.loglog(
        E_TeV,
        results["nubar_CC"],
        label=r"$\bar{\nu}$ CC"
    )

    ax.loglog(
        E_TeV,
        results["nubar_NC"],
        linestyle="--",
        label=r"$\bar{\nu}$ NC"
    )

    # Mark your initial seven energy points
    for E in TEST_ENERGIES_GEV:
        ax.axvline(
            E / 1000.0,
            linestyle=":",
            linewidth=0.7,
            alpha=0.35
        )

    ax.set_xlabel(r"$E_\nu$ [TeV]")
    ax.set_ylabel(
        r"$\sigma_{\nu N}$ [cm$^2$/nucleon]"
    )

    ax.set_title(
        "LO neutrino DIS benchmark using CT18NNLO PDFs"
    )

    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax.set_xlim(0.1, 5.0)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved: {filename}")


# ============================================================
# PRINT VALUES AT YOUR INITIAL ENERGY GRID
# ============================================================

def print_test_grid(results):

    print("\n" + "=" * 78)
    print("CROSS SECTIONS AT THE INITIAL PRODUCTION ENERGY GRID")
    print("=" * 78)

    print(
        f"{'E [TeV]':>10}"
        f"{'nu CC':>18}"
        f"{'nu NC':>18}"
        f"{'nubar CC':>18}"
        f"{'nubar NC':>18}"
    )

    for E in TEST_ENERGIES_GEV:

        i = np.argmin(
            np.abs(results["E_GeV"] - E)
        )

        print(
            f"{results['E_GeV'][i]/1000:10.3f}"
            f"{results['nu_CC'][i]:18.5e}"
            f"{results['nu_NC'][i]:18.5e}"
            f"{results['nubar_CC'][i]:18.5e}"
            f"{results['nubar_NC'][i]:18.5e}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nCalculating theory DIS benchmark...")
    print(f"PDF = {PDF_NAME}, member = {PDF_MEMBER}")
    print(f"Q2 minimum = {Q2_MIN_GEV2} GeV^2")
    print(f"W2 minimum = {W2_MIN_GEV2} GeV^2")
    print()

    results = calculate_all(ENERGIES_GEV)

    save_csv(results)
    print_test_grid(results)
    make_plot(results)
