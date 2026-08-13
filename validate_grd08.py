#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[V1] Validation of the GRD08 melt-viscosity coefficients in magma_viscosity.py
against Giordano, Russell & Dingwell (2008), EPSL 271, 123-134.

Two independent checks back the [V1] clearance (2026-06-20):
  1. Manual: every coefficient (A, b1..b13, c1..c11) AND every oxide grouping was
     checked line-by-line against the paper's Table 1 (mol% oxides) -> exact match.
  2. This script: end-to-end numeric reproduction of the paper's OWN worked
     example, Table 2 -- an iron-free andesite melt + 2.00 wt% H2O, for which the
     paper reports A=-4.55, B=7720, C=334 and log eta = 3.67 at 1273 K.

Run:  python validate_grd08.py     (prints PASS/FAIL, exit 0 = PASS)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from magma_viscosity import grd08_logeta, oxides_to_molpct

# Table 2 hydrous wt% (sums to 100 with 2.00 wt% H2O); F = 0 in this example.
OX = {"SiO2": 61.23, "TiO2": 0.54, "Al2O3": 19.63, "FeOt": 0.03, "MnO": 0.02,
      "MgO": 3.16, "CaO": 8.91, "Na2O": 3.45, "K2O": 0.91, "P2O5": 0.12, "H2O": 2.00}

# Paper's mol% (N) column, for the composition-conversion cross-check.
PAPER_MOLPCT = {"SiO2": 62.38, "TiO2": 0.41, "Al2O3": 11.79, "FeOt": 0.03, "MnO": 0.02,
                "MgO": 4.80, "CaO": 9.73, "Na2O": 3.41, "K2O": 0.59, "P2O5": 0.05, "H2O": 6.80}

T_K = 1273.0
PAPER_LOGETA = 3.67


def cross_check_duplicates():
    """The repo carries TWO hand-typed GRD08 coefficient tables:
    magma_viscosity.grd08_logeta (bulk chain) and giordano2008.log_eta
    (path/overlay chain). Assert they agree to 1e-9 on several hydrous,
    F-bearing compositions so a future edit to one copy cannot silently
    diverge from the other. (Interface differences: F is a positional wt%
    arg vs a dict key 'F'; iron is keyed FeOt vs FeO.)"""
    from giordano2008 import log_eta as g08_log_eta

    comps = [
        # (majors as magma_viscosity keys incl H2O, F_wt, T_C)
        (dict(OX), 0.0, 1000.0),                                        # Table 2 andesite
        (dict(OX, H2O=5.0), 0.45, 800.0),                               # hydrous + F
        (dict(SiO2=63.0, TiO2=0.55, Al2O3=16.5, FeOt=4.0, MnO=0.08,
              MgO=2.2, CaO=4.2, Na2O=3.8, K2O=3.5, P2O5=0.20,
              H2O=10.0), 0.20, 750.0),                                  # ot_majors-like, wet
    ]
    worst = 0.0
    for ox, f_wt, t_c in comps:
        wt = {("FeO" if k == "FeOt" else k): v for k, v in ox.items()}
        wt["F"] = f_wt
        d = abs(grd08_logeta(ox, t_c, f_wt) - g08_log_eta(wt, t_c))
        worst = max(worst, d)
    ok = worst < 1e-9
    print(f"dual-impl x-check : max|dlogeta| = {worst:.2e} over {len(comps)} comps  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    m = oxides_to_molpct(OX, 0.0)
    molpct_ok = all(abs(m[k] - PAPER_MOLPCT[k]) < 0.01 for k in PAPER_MOLPCT)
    lg = grd08_logeta(OX, T_K - 273.15, 0.0)
    eta_ok = abs(lg - PAPER_LOGETA) < 0.02

    print(f"mol% reproduction : {'PASS' if molpct_ok else 'FAIL'}  (max diff "
          f"{max(abs(m[k]-PAPER_MOLPCT[k]) for k in PAPER_MOLPCT):.3f} mol%)")
    print(f"log eta(1273 K)   : {lg:.3f}  (paper {PAPER_LOGETA})  "
          f"-> {'PASS' if eta_ok else 'FAIL'}")
    dup_ok = cross_check_duplicates()
    ok = molpct_ok and eta_ok and dup_ok
    print("[V1] GRD08 validation:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
