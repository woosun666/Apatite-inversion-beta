#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ok Tedi P1 - Lormand apatite volatile FORWARD model (Python port, Tier 1)
=========================================================================
Forward port of the SINGLE-RUN path of `Apatite_V1_SNCC_OR_26214.m`
(Lormand et al. 2024, Lithos; built on Humphreys et al. 2021, EPSL).

Tier 1 = forward model only:
  * volatile evolution along the crystallisation path F (Rayleigh + fluid
    exsolution + fluid salinity), primary boiling OFF;
  * melt mole fractions + OH speciation;
  * non-ideal Cl-OH-F apatite partitioning solved per F step with
    scipy.optimize.least_squares (replaces the MATLAB fmincon active-set
    solve of |x|+|y|+|z| under sum=1; the physical root has x=y=z=0 so the
    sum=1 constraint is satisfied automatically -- checked at runtime).
  * emits the SAME columns as the MATLAB `*_Tgt.csv` for column-wise validation.

NOT ported here (Tier 2/3): the MultiStart/fmincon inversion to observed
apatite ratios, and the diagnostic plotting. For eta(F) see
`magma_viscosity_evolution.py`.

Dependencies: numpy + scipy (+ stdlib csv). No pandas.

Provenance: reproduces a model by C. Lormand, M.C.S. Humphreys et al.
(ERC STEMMS project). For our own research; cite the originals and check
licensing before any public release.

Run (with the MATLAB inputs, validating against the golden _Tgt.csv):
  python apatite_model.py \
    --tf     ".../lormand_2024/1000-790-b1.csv" \
    --golden ".../test_sncc-o-r-26.2.15_test-SNCC-260214_Tgt.csv"
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.optimize import least_squares

# ============================================================================
# PHYSICAL CONSTANTS  (Lormand/Humphreys apatite model; MATLAB ~lines 663-752)
# ----------------------------------------------------------------------------
# The THERMODYNAMIC + STOICHIOMETRIC constants of the model. This is the SINGLE
# SOURCE OF TRUTH: the fast inversion solver in apatite_inversion.py imports
# these (and delta_G() below), so each constant is edited in exactly one place.
# Adjust here to match the Lormand original. NOTE: the FITTED parameters
# (H2O_init, Cl_init, F_init, distribution coefficients, H2O_Sat, bounds, ...)
# are NOT constants -- they live in DEFAULT_PARAMS below and in the
# USER-EDITABLE block of apatite_inversion.py.
# ============================================================================
R_GAS = 8.314                    # gas constant, J/K/mol

# -- Non-ideal apatite mixing: Margules W parameters, kJ/mol  [MATLAB ~707-716]
WG_CLOH = 5.0                    # Wg_ClOH
WG_FOH = 7.0                     # Wg_FOH
WG_FCL = 16.0                    # Wg_FCl
WG_DIFF1 = WG_FOH - WG_FCL       # = -9   enters K_OH_Cl   [MATLAB Wg_diff1]
WG_DIFF2 = WG_CLOH - WG_FCL      # = -11  enters K_OH_F    [MATLAB Wg_diff2]

# -- Standard-state Gibbs energies, dG = A - B*T_K (kJ/mol)  [MATLAB ~731-732]
DG_CLOH_A = 72.9                 # deltaG_ClOH intercept
DG_CLOH_B = 0.034                # deltaG_ClOH slope (per K)
DG_FOH_A = 94.6                  # deltaG_FOH  intercept
DG_FOH_B = 0.04                  # deltaG_FOH  slope (per K)

# -- Molar masses for the melt volatile mole fractions, g/mol  [MATLAB ~696-701]
M_H2O = 18.015
M_OXIDE = 32.0                   # "rest of melt" pseudo-oxide molar mass
M_CL = 35.5
M_F = 19.0

# -- Salinity-block molar masses, g/mol  [MATLAB ~663-674]
M_NACL = 58.44
M_CL_SAL = 35.453                # Cl molar mass used ONLY in the salinity block


def delta_G(T_K):
    """Standard-state dG (kJ/mol) for the Cl-OH and F-OH apatite exchange
    reactions, linear in temperature T_K (Kelvin); MATLAB lines ~731-732.
    Accepts scalar or array T_K. SINGLE SOURCE used by both the apatite_model
    least_squares solve and the apatite_inversion fixed-point solve, so the
    dG coefficients above are defined in one place only.
    Returns (dG_ClOH, dG_FOH)."""
    return DG_CLOH_A - DG_CLOH_B * T_K, DG_FOH_A - DG_FOH_B * T_K

# Output column order = MATLAB WriteData _Tgt.csv (lines ~924-927)
COLUMNS = [
    "Frac_Melt", "K_F_OH_out", "K_Cl_OH_out", "K_Cl_F_out",
    "XF_div_XOH", "XCl_div_XOH", "XF_div_XCl",
    "XH2Ot_m", "XCl_m", "XF_m", "XOH_m",
    "XClap", "XFap", "XOHap",
    "Fluid_inc_mass", "XNaClEq_fluid", "XH2O_Fluid", "NaClagg",
]
# Extra outputs (not in the MATLAB _Tgt.csv): melt volatile concentrations (wt%).
EXTRA_COLUMNS = ["H2O_melt_wtpct", "Cl_melt_wtpct", "F_melt_wtpct"]

# Map MATLAB Multioutput (best-fit) column names -> model parameter keys.
PARAM_MAP = {
    "H2O": "H2O_init", "Cl": "Cl_init", "F": "F_init",
    "Dxtl-mCl": "D_xl_m_Cl", "Dxtl-mF": "D_xl_m_F", "Dxtl-mH2O": "D_xl_m_H2O",
    "Dfl-mCl": "D_fl_m_Cl", "Dfl-mF": "D_fl_m_F", "H2OSat": "H2O_Sat",
}

# Default forward-run parameters = MATLAB CreateTargets block (lines ~106-144)
DEFAULT_PARAMS = dict(
    H2O_init=4.0, Cl_init=0.24, F_init=0.14,
    D_xl_m_Cl=0.2, D_xl_m_F=0.6, D_xl_m_H2O=0.12,
    D_fl_m_Cl=20.0, D_fl_m_F=4.0, H2O_Sat=8.5,
    D_fl_m_PrimBoilH2O=0.0,           # primary boiling OFF
)
K_SPECIATION = 0.361
NUM_FRAC = 200


# ----------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------
def equilibrium(XH2Ot, K_spec):
    """OH mole fraction in the melt from total water (MATLAB `equilibrium`)."""
    a = (K_spec - 4.0) / (2.0 * K_spec)
    inner = 0.25 - (K_spec - 4.0) * (XH2Ot - XH2Ot ** 2) / K_spec
    return (0.5 - np.sqrt(inner)) / a


def volatile_evolution(F, p):
    """First loop of Apatite_Crystallization_Multistart (MATLAB lines ~609-693).

    Concentrations in `p` are wt%; all returned melt/fluid arrays are wt
    FRACTION. Returns melt H2O/Cl/F remaining plus the four salinity columns.
    """
    n = len(F)
    dF = abs(F[1] - F[0])

    H2Ot = np.zeros(n)
    Clf = np.zeros(n)
    Ff = np.zeros(n)
    H2O_rem = np.zeros(n)
    Cl_rem = np.zeros(n)
    F_rem = np.zeros(n)
    fluidH2O_tot = np.zeros(n)
    fluidH2O_inc = np.zeros(n)
    fluidCl = np.zeros(n)
    fluidF = np.zeros(n)
    fluid_inc_mass = np.zeros(n)
    XNaCleq_fluid = np.zeros(n)
    XH2O_fluid = np.zeros(n)
    NaClagg = np.zeros(n)

    H2O_Sat = p["H2O_Sat"]
    DprimB = p["D_fl_m_PrimBoilH2O"]

    # The saturated branch reads [i-1]; if the FIRST step were already saturated,
    # Python's [-1] would silently wrap to the (zero-filled) last element and give
    # wrong values instead of an error. Guard: the path must start undersaturated.
    assert p["H2O_init"] * F[0] ** (p["D_xl_m_H2O"] - 1.0) <= p["H2O_Sat"], (
        f"first step already H2O-saturated (H2O_init={p['H2O_init']}, "
        f"H2O_Sat={p['H2O_Sat']}): [i-1] indexing invalid")

    for i in range(n):
        H2Ot[i] = (p["H2O_init"] * F[i] ** (p["D_xl_m_H2O"] - 1.0)) * 0.01

        if H2Ot[i] <= H2O_Sat * 0.01:
            # ---- undersaturated: pure Rayleigh ----
            Clf[i] = (p["Cl_init"] * F[i] ** (p["D_xl_m_Cl"] - 1.0)) * 0.01
            Ff[i] = (p["F_init"] * F[i] ** (p["D_xl_m_F"] - 1.0)) * 0.01
            H2O_rem[i] = H2Ot[i] - fluidH2O_tot[i]      # fluidH2O_tot[i] == 0
            Cl_rem[i] = Clf[i] - fluidCl[i]
            F_rem[i] = Ff[i] - fluidF[i]
        else:
            # ---- saturated: H2O buffered, fluid exsolves ----
            Clf[i] = Cl_rem[i - 1] * (F[i] / F[i - 1]) ** (p["D_xl_m_Cl"] - 1.0)
            Ff[i] = F_rem[i - 1] * (F[i] / F[i - 1]) ** (p["D_xl_m_F"] - 1.0)

            fluidH2O_tot[i] = (p["H2O_init"] * F[i] ** (p["D_xl_m_H2O"] - 1.0)) - H2O_Sat
            fluidH2O_inc[i] = (fluidH2O_tot[i] - fluidH2O_tot[i - 1]) / 100.0

            fluidCl[i] = (p["D_fl_m_Cl"] * Clf[i]) * fluidH2O_inc[i]
            fluidF[i] = (p["D_fl_m_F"] * Ff[i]) * fluidH2O_inc[i]

            # ---- salinity ----
            fluid_inc_mass[i] = fluidCl[i] + fluidH2O_inc[i]
            mass_NaCl = fluidCl[i] * M_NACL / M_CL_SAL
            Cl_fluid_wt = fluidCl[i] / (fluidH2O_inc[i] + fluidCl[i])
            NaCleq_inst = Cl_fluid_wt * (M_NACL / M_CL_SAL)
            denom_sal = (mass_NaCl / M_NACL) + (fluidH2O_inc[i] / M_H2O)
            XNaCleq_fluid[i] = (mass_NaCl / M_NACL) / denom_sal
            XH2O_fluid[i] = (fluidH2O_inc[i] / M_H2O) / denom_sal
            NaClagg[i] = NaCleq_inst * fluid_inc_mass[i]

            H2O_rem[i] = H2Ot[i] - (fluidH2O_tot[i] / 100.0) - (dF * DprimB)
            H2O_Sat = H2O_Sat - (100.0 * dF * DprimB)        # unchanged if DprimB==0
            Cl_rem[i] = Clf[i] - fluidCl[i]
            F_rem[i] = Ff[i] - fluidF[i]

    return dict(
        H2O_rem=H2O_rem, Cl_rem=Cl_rem, F_rem=F_rem,
        Fluid_inc_mass=fluid_inc_mass, XNaClEq_fluid=XNaCleq_fluid,
        XH2O_Fluid=XH2O_fluid, NaClagg=NaClagg,
    )


def melt_mole_fractions(H2O_rem, Cl_rem, F_rem, K_spec):
    """Melt volatile mole fractions + OH speciation (MATLAB lines ~696-702)."""
    denom = (H2O_rem / M_H2O) + ((1.0 - H2O_rem) / M_OXIDE)
    XH2Ot = (H2O_rem / M_H2O) / denom
    XCl = (Cl_rem / M_CL) / denom
    XF = (F_rem / M_F) / denom
    XOH = equilibrium(XH2Ot, K_spec)
    return XH2Ot, XCl, XF, XOH


def _kd_factories(T_K):
    """Return K_OH_Cl(.) and K_OH_F(.) as functions of apatite composition
    at temperature T_K (MATLAB anonymous functions, lines ~743-752)."""
    dG_ClOH, dG_FOH = delta_G(T_K)

    def K_OH_Cl(XClap, XOHap, XFap):
        return np.exp(
            (1000.0 * (-dG_ClOH - ((XClap - XOHap) * WG_CLOH + XFap * WG_DIFF1)))
            / (R_GAS * T_K)
        )

    def K_OH_F(XClap, XOHap, XFap):
        return np.exp(
            1000.0 * (-dG_FOH - ((XFap - XOHap) * WG_FOH + XClap * WG_DIFF2))
            / (R_GAS * T_K)
        )

    return K_OH_Cl, K_OH_F


def solve_partition_point(XF_m, XCl_m, XOH_m, T_K, x0=(0.7, 0.1, 0.2)):
    """Solve non-ideal Cl-OH-F apatite partitioning at one F step.

    Unknowns v = [XClap, XOHap, XFap]; residuals are the three coupled
    equations of GetApMoleFractions (MATLAB lines ~861-885). The KDs depend
    on apatite composition, so this is an implicit 3-equation root find.
    """
    K_OH_Cl, K_OH_F = _kd_factories(T_K)

    def resid(v):
        XClap, XOHap, XFap = v
        k_oh_cl = K_OH_Cl(XClap, XOHap, XFap)
        k_oh_f = K_OH_F(XClap, XOHap, XFap)
        Kd_ClF = k_oh_f / k_oh_cl
        Kd_FOH = 1.0 / k_oh_f
        Kd_ClOH = 1.0 / k_oh_cl
        x = XFap - 1.0 / (1.0 + (XCl_m * Kd_ClF / XF_m) + (XOH_m / (XF_m * Kd_FOH)))
        y = XClap - 1.0 / (1.0 + (XF_m / (XCl_m * Kd_ClF)) + (XOH_m / (XCl_m * Kd_ClOH)))
        z = XOHap - 1.0 / (1.0 + (XF_m * Kd_FOH / XOH_m) + (XCl_m * Kd_ClOH / XOH_m))
        return [x, y, z]

    sol = least_squares(resid, x0, bounds=([0, 0, 0], [1, 1, 1]),
                        xtol=1e-13, ftol=1e-13, gtol=1e-13)
    XClap, XOHap, XFap = sol.x
    k_oh_cl = K_OH_Cl(XClap, XOHap, XFap)
    k_oh_f = K_OH_F(XClap, XOHap, XFap)
    K_Cl_OH = 1.0 / k_oh_cl
    K_F_OH = 1.0 / k_oh_f
    K_Cl_F = k_oh_f / k_oh_cl
    return XClap, XOHap, XFap, K_Cl_OH, K_F_OH, K_Cl_F, abs(sol.x.sum() - 1.0)


def forward_model(F, T_K_vec, params=DEFAULT_PARAMS, K_spec=K_SPECIATION):
    """Full single-run forward model -> (dict of column arrays, max_sum_err)."""
    vol = volatile_evolution(F, params)
    XH2Ot, XCl_m, XF_m, XOH_m = melt_mole_fractions(
        vol["H2O_rem"], vol["Cl_rem"], vol["F_rem"], K_spec)

    n = len(F)
    XClap = np.zeros(n); XFap = np.zeros(n); XOHap = np.zeros(n)
    K_Cl_OH = np.zeros(n); K_F_OH = np.zeros(n); K_Cl_F = np.zeros(n)
    max_sum_err = 0.0
    for j in range(n):
        cl, oh, f, kclo, kfo, kclf, sum_err = solve_partition_point(
            XF_m[j], XCl_m[j], XOH_m[j], T_K_vec[j])
        XClap[j], XOHap[j], XFap[j] = cl, oh, f
        K_Cl_OH[j], K_F_OH[j], K_Cl_F[j] = kclo, kfo, kclf
        max_sum_err = max(max_sum_err, sum_err)

    data = {
        "Frac_Melt": np.asarray(F, float),
        "K_F_OH_out": K_F_OH, "K_Cl_OH_out": K_Cl_OH, "K_Cl_F_out": K_Cl_F,
        "XF_div_XOH": XFap / XOHap, "XCl_div_XOH": XClap / XOHap,
        "XF_div_XCl": XFap / XClap,
        "XH2Ot_m": XH2Ot, "XCl_m": XCl_m, "XF_m": XF_m, "XOH_m": XOH_m,
        "XClap": XClap, "XFap": XFap, "XOHap": XOHap,
        "Fluid_inc_mass": vol["Fluid_inc_mass"],
        "XNaClEq_fluid": vol["XNaClEq_fluid"],
        "XH2O_Fluid": vol["XH2O_Fluid"], "NaClagg": vol["NaClagg"],
        # melt volatile concentrations (wt%) -- the quantification output
        "H2O_melt_wtpct": vol["H2O_rem"] * 100.0,
        "Cl_melt_wtpct": vol["Cl_rem"] * 100.0,
        "F_melt_wtpct": vol["F_rem"] * 100.0,
    }
    return data, max_sum_err


# ----------------------------------------------------------------------------
# Temperature handling
# ----------------------------------------------------------------------------
def build_F_and_T(tf_path=None, const_T_C=None, num_frac=NUM_FRAC):
    """Reproduce the MATLAB T-F handling.

    With a TF curve (csv cols Var1=T_C, Var2=F): F_start=max(F),
    F_end=max(0.1,min(F)); F = linspace; T(F) by linear interpolation +273.15.
    Without it: fixed temperature, F from 1 -> 0.1.
    """
    if tf_path and os.path.exists(tf_path):
        tf = np.genfromtxt(tf_path, delimiter=",", names=True)
        f_nodes = np.asarray(tf["Var2"], float)
        t_nodes = np.asarray(tf["Var1"], float)
        # unique on F, keep first (MATLAB unique(...,'stable'))
        _, keep = np.unique(f_nodes, return_index=True)
        keep = np.sort(keep)
        f_nodes, t_nodes = f_nodes[keep], t_nodes[keep]
        F_start = f_nodes.max()
        F_end = max(0.1, f_nodes.min())
        F = np.linspace(F_start, F_end, num_frac)
        order = np.argsort(f_nodes)           # np.interp needs ascending x
        T_K = np.interp(F, f_nodes[order], t_nodes[order]) + 273.15
        return F, T_K
    if const_T_C is None:
        raise ValueError("provide --tf <curve.csv> or --const-T <degC>")
    F = np.linspace(1.0, 0.1, num_frac)
    T_K = np.full(num_frac, const_T_C + 273.15)
    return F, T_K


def generate_tf_curve(T_initial_C, T_final_C, b, num_frac=NUM_FRAC):
    """T-F cooling path from the Huber et al. (2010) parameterization

        F = ((T - T_final) / (T_initial - T_final)) ** b

    (F = melt fraction 0-1, b = cooling coefficient), evaluated analytically
    on the F grid via the inverse T(F) = T_final + (T_initial - T_final) *
    F**(1/b) -- monotone for b > 0, so no numeric root-finding.

    T_initial / T_final are the EFFECTIVE start / end temperatures of the
    modeled crystallization interval, NOT the liquidus / solidus of a hydrous
    granitic melt: for the OT runs T_initial ~1000 degC is the onset of
    fractional crystallization of a mafic-to-intermediate parental magma at
    ~15 km depth (Collins et al. 2021), and T_final is the near-solidus end
    (~790 degC for the PMD end-member per Ti-in-zircon thermometry, Schiller
    & Finger 2019, and experimental constraints, Zhang et al. 2020, Xiao et
    al. 2024, Grocolas et al. 2025).

    Same grid + Kelvin conventions as build_F_and_T (F from 1 -> 0.1,
    T_K = T_C + 273.15); b=1 / b=2 reproduce the digitized 1000-750-b1 /
    ...-B2 curves used so far.
    """
    T_i, T_f, b = float(T_initial_C), float(T_final_C), float(b)
    if not T_i > T_f:
        raise ValueError(f"need T_initial > T_final (got {T_i} <= {T_f})")
    if not b > 0:
        raise ValueError(f"need cooling coefficient b > 0 (got {b})")
    F = np.linspace(1.0, 0.1, num_frac)
    T_K = T_f + (T_i - T_f) * F ** (1.0 / b) + 273.15
    return F, T_K


def load_params_from_multioutput(path, row=0):
    """Read one best-fit parameter set from a MATLAB Multioutput_*.csv
    (rows sorted by ascending RMSE; row 0 = best fit). Returns (params, rmse).
    D_fl_m_PrimBoilH2O is not a fitted parameter -> taken from DEFAULT_PARAMS."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no rows in {path}")
    r = rows[row]
    params = dict(DEFAULT_PARAMS)               # carries D_fl_m_PrimBoilH2O
    for src, dst in PARAM_MAP.items():
        params[dst] = float(r[src])
    rmse = float(r.get("RMSE", "nan"))
    return params, rmse


# ----------------------------------------------------------------------------
# I/O + validation
# ----------------------------------------------------------------------------
def write_csv(data, path):
    n = len(data["Frac_Melt"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = COLUMNS + EXTRA_COLUMNS
        w.writerow(cols)
        for i in range(n):
            w.writerow(["%.15g" % float(data[c][i]) for c in cols])


def compare_to_golden(data, golden_path):
    """Column-wise max-abs / max-rel difference vs the MATLAB _Tgt.csv."""
    gold = np.genfromtxt(golden_path, delimiter=",", names=True)
    gcols = set(gold.dtype.names)
    n = min(len(data["Frac_Melt"]), gold.shape[0])
    print(f"\n[validation] vs {os.path.basename(golden_path)}  (comparing {n} rows)")
    print(f"{'column':>16} {'max|abs|':>12} {'max|rel|':>12}")
    print("-" * 42)
    worst = 0.0
    for c in COLUMNS:
        if c not in gcols:
            print(f"{c:>16} {'(missing)':>12}")
            continue
        a = np.asarray(data[c], float)[:n]
        b = np.asarray(gold[c], float)[:n]
        abs_d = np.nanmax(np.abs(a - b))
        scale = np.nanmax(np.abs(b))
        rel_d = abs_d / scale if scale > 0 else 0.0
        worst = max(worst, rel_d)
        print(f"{c:>16} {abs_d:12.3e} {rel_d:12.3e}")
    print("-" * 42)
    print(f"worst column max|rel| = {worst:.3e}  -> "
          f"{'PASS (<1e-4)' if worst < 1e-4 else 'CHECK'}")
    return worst


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Lormand apatite forward model (Python, Tier 1)")
    ap.add_argument("--tf", default=None, help="TF curve csv (Var1=T_C, Var2=F)")
    ap.add_argument("--const-T", type=float, default=None,
                    help="constant T in degC if no TF curve")
    ap.add_argument("--params", default=None,
                    help="MATLAB Multioutput_*.csv of best-fit params (else CreateTargets defaults)")
    ap.add_argument("--params-row", type=int, default=0,
                    help="row in --params to use (0 = lowest RMSE / best fit)")
    ap.add_argument("--num-frac", type=int, default=NUM_FRAC)
    ap.add_argument("--out", default=None, help="output csv path")
    ap.add_argument("--golden", default=None, help="MATLAB _Tgt.csv to validate against")
    args = ap.parse_args()

    if args.params:
        params, rmse = load_params_from_multioutput(args.params, args.params_row)
        print(f"[params] best-fit row {args.params_row} from "
              f"{os.path.basename(args.params)} (RMSE={rmse:.4g})")
        for k in ("H2O_init", "Cl_init", "F_init", "H2O_Sat"):
            print(f"         {k:10s} = {params[k]:.4g}")
    else:
        params = DEFAULT_PARAMS
        print("[params] MATLAB CreateTargets defaults (target curve)")

    F, T_K = build_F_and_T(args.tf, args.const_T, args.num_frac)
    print(f"[input] F: {F[0]:.3f} -> {F[-1]:.3f} ({len(F)} steps); "
          f"T: {T_K[0]-273.15:.1f} -> {T_K[-1]-273.15:.1f} degC")

    data, max_sum_err = forward_model(F, T_K, params)
    print(f"[solve] max |sum(Xap)-1| over all steps = {max_sum_err:.2e} "
          f"(sum=1 constraint satisfied by the physical root)")

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "apatite_forward_python_Tgt.csv")
    write_csv(data, out)
    print(f"[written] {out}")

    print(f"[melt H2O/Cl/F wt%] start: {data['H2O_melt_wtpct'][0]:.2f} / "
          f"{data['Cl_melt_wtpct'][0]:.3f} / {data['F_melt_wtpct'][0]:.3f}   "
          f"end: {data['H2O_melt_wtpct'][-1]:.2f} / {data['Cl_melt_wtpct'][-1]:.3f} / "
          f"{data['F_melt_wtpct'][-1]:.3f}")

    if args.golden and os.path.exists(args.golden):
        compare_to_golden(data, args.golden)


if __name__ == "__main__":
    main()
