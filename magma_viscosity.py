#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ok Tedi P1 — bulk magma viscosity calculator
=============================================
Pipeline:   eta_magma = eta_melt(GRD08) * eta_r_crystal * eta_r_bubble

  1) eta_melt  : Giordano, Russell & Dingwell (2008) EPSL  -> liquid (melt) viscosity
  2) crystals  : Roscoe/Einstein-Roscoe (default) OR Costa et al. (2009)  -> bulk magma
  3) bubbles   : Llewellin & Manga (2005), two regimes (pre- vs post-exsolution)

------------------------------------------------------------------------------
HONESTY / VERIFY-BEFORE-PUBLISH FLAGS  (do not skip)
------------------------------------------------------------------------------
[V1] VERIFIED 2026-06-20. The coefficient table below (A, b1..b13, c1..c11) and
     every oxide grouping were checked line-by-line against Giordano et al. (2008)
     Table 1 (mol% oxides): exact match. End-to-end numeric check against the
     paper's own worked example (Table 2: iron-free andesite + 2.00 wt% H2O) ->
     this code gives log eta = 3.675 at 1273 K vs the paper's 3.67, mol% column
     reproduced exactly. Re-run: python validate_grd08.py.
[V2] F is entered as wt% F and converted to the GRD08 "F2O-1" component
     (n_F2O-1 = (wt%F / 18.998) / 2). Confirm this conversion against the paper.
     Set F = 0 to obtain an H2O-only estimate as a cross-check.
[V3] Whole-rock != melt. A crystal-rich porphyry whole-rock is NOT the liquid
     composition; using it as the melt in GRD08 biases eta_melt. This is an
     explicit modelling choice (option A) and must be stated as a caveat.
[V4] Costa (2009) parameters are intentionally left as None — fill from the
     original paper if you use that model. The default Roscoe constants
     (B=2.5, phi_max=0.6; Roscoe 1952) are well established and run as-is.
[V5] Bubble end-members (LM2005): low capillary number Ca<<1 -> (1-phi_b)^-1
     (viscosity UP); high Ca>>1 -> (1-phi_b)^(5/3) (viscosity DOWN). Pick the
     regime appropriate to your strain rate; both are printed.
------------------------------------------------------------------------------
"""

import math

# =============================================================================
# 1) GRD08 melt viscosity
# =============================================================================

# Oxide molar masses (g/mol). F handled separately as F2O-1.
M = {
    "SiO2": 60.084, "TiO2": 79.866, "Al2O3": 101.961, "FeOt": 71.844,
    "MnO": 70.937, "MgO": 40.304, "CaO": 56.077, "Na2O": 61.979,
    "K2O": 94.196, "P2O5": 141.945, "H2O": 18.015,
}
M_F = 18.998  # atomic F

# ---- GRD08 Table 1 coefficients — [V1] VERIFIED 2026-06-20 vs Giordano et al. (2008) ----
A_GRD = -4.55
B = {  # B-block coefficients
    "b1": 159.6, "b2": -173.3, "b3": 72.1, "b4": 75.7, "b5": -39.0,
    "b6": -84.1, "b7": 141.5, "b11": -2.43, "b12": -0.91, "b13": 17.6,
}
C = {  # C-block coefficients
    "c1": 2.75, "c2": 15.7, "c3": 8.3, "c4": 10.2, "c5": -12.3,
    "c6": -99.5, "c11": 0.30,
}
# --------------------------------------------------------------------------------

def oxides_to_molpct(ox, F_wt=0.0):
    """ox: dict wt% of the M-keys (incl. H2O). F_wt: wt% F. Returns mol% dict
    over GRD08 components incl. H2O and F2O-1."""
    n = {k: (ox.get(k, 0.0) / M[k]) for k in M}
    n["F2O-1"] = (F_wt / M_F) / 2.0          # [V2] confirm against paper
    tot = sum(n.values())
    return {k: 100.0 * v / tot for k, v in n.items()}

def grd08_logeta(ox, T_C, F_wt=0.0):
    """Melt (liquid) log10 viscosity in Pa.s at T (deg C)."""
    m = oxides_to_molpct(ox, F_wt)
    SiO2, TiO2, Al2O3 = m["SiO2"], m["TiO2"], m["Al2O3"]
    FeOt, MnO, MgO    = m["FeOt"], m["MnO"], m["MgO"]
    CaO, Na2O, K2O    = m["CaO"], m["Na2O"], m["K2O"]
    P2O5, H2O, FF     = m["P2O5"], m["H2O"], m["F2O-1"]

    Bcalc = (
        B["b1"]*(SiO2 + TiO2)
        + B["b2"]*(Al2O3)
        + B["b3"]*(FeOt + MnO + P2O5)
        + B["b4"]*(MgO)
        + B["b5"]*(CaO)
        + B["b6"]*(Na2O + H2O + FF)
        + B["b7"]*(H2O + FF + math.log(1.0 + H2O))
        + B["b11"]*((SiO2 + TiO2)*(FeOt + MnO + MgO))
        + B["b12"]*((SiO2 + TiO2 + Al2O3 + P2O5)*(Na2O + K2O + H2O))
        + B["b13"]*((Al2O3)*(Na2O + K2O))
    )
    Ccalc = (
        C["c1"]*(SiO2)
        + C["c2"]*(TiO2 + Al2O3)
        + C["c3"]*(FeOt + MnO + MgO)
        + C["c4"]*(CaO)
        + C["c5"]*(Na2O + K2O)
        + C["c6"]*(math.log(1.0 + H2O + FF))
        + C["c11"]*((Al2O3 + FeOt + MnO + MgO + CaO - P2O5)*(Na2O + K2O + H2O + FF))
    )
    T_K = T_C + 273.15
    return A_GRD + Bcalc / (T_K - Ccalc)      # log10 eta (Pa.s)

# =============================================================================
# 2) crystal correction  -> bulk magma
# =============================================================================

def eta_r_roscoe(phi, phi_max=0.60, Bexp=2.5):
    """[V4] Roscoe/Einstein-Roscoe relative viscosity."""
    if phi >= phi_max:
        return float("inf")
    return (1.0 - phi / phi_max) ** (-Bexp)

# Costa et al. (2009) — fill params from the paper to enable.
COSTA = {"phi_star": None, "xi": None, "gamma": None, "delta": None, "B": None}
def eta_r_costa(phi):
    if any(v is None for v in COSTA.values()):
        raise ValueError("Costa(2009) parameters not set — fill COSTA from the paper [V4].")
    ps, xi, ga, de, Bc = (COSTA[k] for k in ("phi_star","xi","gamma","delta","B"))
    x = phi / ps
    num = 1.0 + x ** de
    erf_arg = (math.sqrt(math.pi) / 2.0) * x / xi * (1.0 + x ** ga)
    den = (1.0 - xi * math.erf(erf_arg)) ** (Bc * ps)
    return num / den

# =============================================================================
# 3) bubble correction  (Llewellin & Manga 2005) — [V5]
# =============================================================================

def eta_r_bubble(phi_b, regime="low_Ca"):
    if phi_b <= 0.0:
        return 1.0
    if regime == "low_Ca":      # spherical bubbles, viscosity UP
        return (1.0 - phi_b) ** (-1.0)
    elif regime == "high_Ca":   # deformed bubbles, viscosity DOWN
        return (1.0 - phi_b) ** (5.0 / 3.0)
    raise ValueError("regime must be 'low_Ca' or 'high_Ca'")

# =============================================================================
# driver
# =============================================================================

def report(ox, T_C, H2O_list, F_wt, phi_xtl, phi_bub_post,
           crystal_model="roscoe"):
    print("="*72)
    print("Ok Tedi bulk magma viscosity  (GRD08 x crystals x bubbles)")
    print("="*72)
    print(f"T = {T_C:.0f} C | F = {F_wt:.3f} wt% | crystal phi = {phi_xtl:.2f} "
          f"| post-exsolution bubble phi = {phi_bub_post:.2f}")
    print(f"crystal model = {crystal_model}")
    print("-"*72)
    print(f"{'H2O wt%':>8} | {'log eta_melt':>12} | {'log eta_bulk':>12} "
          f"| {'pre-exsol':>10} | {'post low_Ca':>11} | {'post high_Ca':>12}")
    print("-"*72)
    for w in H2O_list:
        ox_w = dict(ox); ox_w["H2O"] = w
        lg_melt = grd08_logeta(ox_w, T_C, F_wt)
        if crystal_model == "roscoe":
            rx = eta_r_roscoe(phi_xtl)
        else:
            rx = eta_r_costa(phi_xtl)
        lg_bulk = lg_melt + math.log10(rx)                       # crystals only
        lg_pre  = lg_bulk + math.log10(eta_r_bubble(0.0))        # bubbles negligible
        lg_lowCa  = lg_bulk + math.log10(eta_r_bubble(phi_bub_post, "low_Ca"))
        lg_highCa = lg_bulk + math.log10(eta_r_bubble(phi_bub_post, "high_Ca"))
        print(f"{w:8.2f} | {lg_melt:12.2f} | {lg_bulk:12.2f} "
              f"| {lg_pre:10.2f} | {lg_lowCa:11.2f} | {lg_highCa:12.2f}")
    print("-"*72)
    print("values are log10(viscosity / Pa.s). pre-exsolution: bubbles ~ absent.")
    print("post-exsolution: low_Ca = viscosity increase; high_Ca = decrease (LM2005).")


def self_test(ox, F_wt):
    """[structural check] as T -> very high, log eta_melt -> A = -4.55."""
    lg = grd08_logeta(dict(ox, H2O=2.0), 1e7, F_wt)
    ok = abs(lg - A_GRD) < 0.02
    print(f"[self-test] high-T limit log eta = {lg:.3f} (expect {A_GRD}) -> "
          f"{'PASS' if ok else 'CHECK'}\n")


def sweep_report(ox, T_C, H2O_paths, F_wt, phi_list, phi_bub_post,
                 crystal_model="roscoe", bubble_regime="low_Ca"):
    """Auto table: crystal phi (rows) x H2O paths (cols). Each cell:
    bulk(pre-exsolution) / bulk(post-exsolution) log10 viscosity."""
    print("="*72)
    print(f"Bulk magma viscosity sweep | T={T_C:.0f} C | F={F_wt} wt% "
          f"| {crystal_model} | post-exsol bubble phi_b={phi_bub_post} ({bubble_regime})")
    hdr = f"{'phi':>6} |" + "".join(f"{('H2O='+format(w,'g')+'wt%'):>16}|" for w in H2O_paths)
    print("-"*len(hdr)); print(hdr); print("-"*len(hdr))
    for phi in phi_list:
        rx = eta_r_roscoe(phi) if crystal_model == "roscoe" else eta_r_costa(phi)
        row = f"{phi:6.2f} |"
        for w in H2O_paths:
            lgm = grd08_logeta(dict(ox, H2O=w), T_C, F_wt)
            lgb = lgm + math.log10(rx)                                   # pre-exsol
            lgp = lgb + math.log10(eta_r_bubble(phi_bub_post, bubble_regime))
            row += f"   {lgb:5.2f}/{lgp:5.2f} |"
        print(row)
    print("-"*len(hdr))
    print("cells = log10(eta/Pa.s): bulk pre-exsol / bulk post-exsol.")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # PLACEHOLDER whole-rock majors (generic andesite-dacite, anhydrous wt%)
    # — NOT an Ok Tedi analysis; same demo values as giordano2008.ot_majors.
    # Replace with your own least-altered whole-rock composition (FeOt =
    # total Fe as FeO). Mol-fraction conversion normalises internally.
    # -------------------------------------------------------------------------
    whole_rock = {   # PLACEHOLDER demo majors, anhydrous wt%
        "SiO2": 63.0, "TiO2": 0.55, "Al2O3": 16.5, "FeOt": 4.0, "MnO": 0.08,
        "MgO": 2.2, "CaO": 4.2, "Na2O": 3.8, "K2O": 3.5, "P2O5": 0.20,
    }
    T_C          = 800.0            # Ti-in-zircon, mobile ore-forming magma state
    H2O_paths    = [5.0, 10.0]      # undersaturated ~5 ; saturated ~10 (10 > GRD08 calib, extrapolation)
    F_wt         = 0.20             # apatite-derived melt F (wt%)
    phi_list     = [0.10, 0.20, 0.30, 0.40]   # mobile: keep below ~0.5 lock-up
    phi_bub_post = 0.10             # post-exsolution bubble fraction (second-order)

    self_test(whole_rock, F_wt)
    sweep_report(whole_rock, T_C, H2O_paths, F_wt, phi_list, phi_bub_post,
                 crystal_model="roscoe", bubble_regime="low_Ca")
