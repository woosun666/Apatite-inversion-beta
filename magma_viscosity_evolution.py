#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ok Tedi P1 — bulk magma viscosity AS A FUNCTION OF MELT FRACTION  eta(F)
========================================================================
Extension of `magma_viscosity.py` that walks the melt fraction F (Lormand's
crystallisation variable) and, at every step, evaluates the bulk magma viscosity:

    eta_magma(F) = eta_melt(GRD08; comp(F), H2O(F), F_halogen(F), T(F))
                   x eta_r_crystal( phi(F) )           [Roscoe]
                   x eta_r_bubble ( phi_b(F) )         [Llewellin & Manga 2005]

What evolves with F, and where it comes from
--------------------------------------------
  * H2O(F), F_halogen(F)  : reproduced from the LORMAND volatile-evolution
                            equations (Rayleigh fractionation + fluid exsolution
                            buffering at H2O_Sat). Same equations as
                            Apatite_V1_SNCC_OR_26214.m, primary boiling OFF.
  * major oxides(F), T(F) : read from a rhyolite-MELTS LIQUID LINE OF DESCENT
                            (LLD) table -> this is the scientifically rigorous
                            source the scope (Option A) requires. GRD08 melt
                            viscosity is DOMINATED by the major-element melt
                            chemistry, which Lormand's model does not track.
  * phi(F) = 1 - F        : crystal fraction is the complement of melt fraction.
                            Roscoe diverges at phi_max (~0.6) => rheological
                            LOCK-UP; the mobile window (scope: phi < 0.5) is F > 0.5.
  * phi_b(F)              : 0 while undersaturated; phi_bub_post once H2O>H2O_Sat.

HONESTY / VERIFY-BEFORE-PUBLISH (inherited + new)
-------------------------------------------------
[V1] GRD08 coefficients are reused from magma_viscosity.py (single source of
     truth); VERIFIED 2026-06-20 there against Giordano et al. (2008) Table 1
     (see validate_grd08.py).
[V6] The supplied MELTS table is the controlling input. The bundled
     example_melts_lld.csv is ILLUSTRATIVE ONLY (not Ok Tedi). Numbers are not
     reportable until a real 24OT02-57 MELTS LLD is plugged in.
[V7] eta(F) is a PROPERTY TRAJECTORY (how viscosity evolves through
     crystallisation/degassing), reported as an intensive property only. It is
     NOT an ascent/transport/flux argument (that is P3; scope sec.10).
"""

import csv
import math
import os
import sys

# Reuse the verified GRD08 + correction functions (single source of truth).
from magma_viscosity import grd08_logeta, eta_r_roscoe, eta_r_bubble

HERE = os.path.dirname(os.path.abspath(__file__))

# Oxides expected from the MELTS LLD (anhydrous; renormalised internally).
MAJ_OXIDES = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO",
              "MgO", "CaO", "Na2O", "K2O", "P2O5"]


# =============================================================================
# 1) Lormand volatile evolution H2O(F), F_halogen(F)  (primary boiling OFF)
#    Mirrors Apatite_V1_SNCC_OR_26214.m lines ~597-674.
# =============================================================================

def lormand_volatiles(F_grid, p):
    """Return per-F melt H2O (wt%) and melt F (wt%) along the crystallisation
    path, plus a boolean 'saturated' flag. p is a dict of Lormand parameters.

    Melt H2O follows Rayleigh until it reaches H2O_Sat, then is buffered at
    H2O_Sat as excess water exsolves to a fluid (primary boiling off).
    Melt F follows Rayleigh until saturation, then loses F to the fluid via
    D_fl_m_F weighted by the incremental exsolved-water mass.
    """
    H2O_init = p["H2O_init"]
    F_init = p["F_init"]
    D_H2O = p["D_xl_m_H2O"]
    D_F = p["D_xl_m_F"]
    D_fl_m_F = p["D_fl_m_F"]
    H2O_Sat = p["H2O_Sat"]

    n = len(F_grid)
    H2O_melt = [0.0] * n        # wt% water remaining in melt
    Fmelt_rem = [0.0] * n       # wt% F remaining in melt
    saturated = [False] * n

    fluid_H2O_tot_prev = 0.0    # wt% water cumulatively exsolved
    for i, Fr in enumerate(F_grid):
        H2Ot = H2O_init * Fr ** (D_H2O - 1.0)          # bulk-path melt water (wt%)
        if H2Ot <= H2O_Sat:
            # ---- undersaturated: pure Rayleigh ----
            H2O_melt[i] = H2Ot
            Fmelt = F_init * Fr ** (D_F - 1.0)
            Fmelt_rem[i] = Fmelt
            fluid_H2O_tot_i = 0.0
        else:
            # ---- saturated: water buffered, fluid exsolves ----
            saturated[i] = True
            H2O_melt[i] = H2O_Sat                       # buffered (prim. boiling off)
            fluid_H2O_tot_i = H2Ot - H2O_Sat            # wt% cumulatively exsolved
            inc_H2O = (fluid_H2O_tot_i - fluid_H2O_tot_prev) / 100.0  # wt frac this step
            # melt F continues to fractionate from the previous remaining value
            if i > 0 and Fmelt_rem[i - 1] > 0:
                Fmelt = Fmelt_rem[i - 1] * (Fr / F_grid[i - 1]) ** (D_F - 1.0)
            else:
                Fmelt = F_init * Fr ** (D_F - 1.0)
            fluid_F_frac = (D_fl_m_F * Fmelt) * inc_H2O
            Fmelt_rem[i] = max(Fmelt - fluid_F_frac, 0.0)
        fluid_H2O_tot_prev = fluid_H2O_tot_i
    return H2O_melt, Fmelt_rem, saturated


# =============================================================================
# 2) MELTS liquid line of descent: read + interpolate onto the F grid
# =============================================================================

def load_melts_lld(path):
    """Read a MELTS LLD csv. Lines starting with '#' are comments. Returns the
    node F vector (descending-sorted) and a dict of node arrays (T_C + oxides).
    Header names are matched case-insensitively; H2O/F columns, if any, are
    ignored (volatiles come from Lormand)."""
    def _is_comment(ln):
        s = ln.lstrip()
        if s[:1] in ('"', "'"):
            s = s[1:].lstrip()
        return s.startswith("#")

    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh if ln.strip() and not _is_comment(ln)]
    reader = csv.DictReader(lines)
    # case-insensitive column map
    colmap = {c.lower().strip(): c for c in reader.fieldnames}

    def col(name):
        return colmap.get(name.lower())

    need = ["F", "T_C"] + MAJ_OXIDES
    missing = [k for k in need if col(k) is None]
    if missing:
        raise ValueError("MELTS LLD missing columns: %s\n  found: %s"
                         % (missing, reader.fieldnames))
    for r in reader:
        rows.append({k: float(r[col(k)]) for k in need})
    rows.sort(key=lambda d: d["F"], reverse=True)     # F from 1 -> low
    out = {"F": [r["F"] for r in rows]}
    for k in ["T_C"] + MAJ_OXIDES:
        out[k] = [r[k] for r in rows]
    return out


def _interp(xq, xs, ys):
    """Linear interpolation with flat extrapolation. xs ASCENDING."""
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xq <= xs[i]:
            t = (xq - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def melts_at_F(lld, Fq):
    """Interpolated (T_C, anhydrous-renormalised oxide dict) at melt fraction Fq."""
    # interpolation needs ascending x
    Fs = lld["F"][::-1]
    T = _interp(Fq, Fs, lld["T_C"][::-1])
    ox = {k: _interp(Fq, Fs, lld[k][::-1]) for k in MAJ_OXIDES}
    s = sum(ox.values())
    ox = {k: 100.0 * v / s for k, v in ox.items()}     # anhydrous -> 100 wt%
    return T, ox


# =============================================================================
# 3) eta(F) driver
# =============================================================================

def viscosity_evolution(lld, p, F_grid,
                        phi_max=0.60, Bexp=2.5,
                        phi_bub_post=0.10, bubble_regime="low_Ca",
                        phi_override=None):
    """Return a list of per-F dicts with the full viscosity breakdown.
    phi(F)=1-F by default; pass phi_override(list) to fix crystallinity instead."""
    H2O_melt, Fmelt, sat = lormand_volatiles(F_grid, p)
    out = []
    for i, Fr in enumerate(F_grid):
        T_C, ox = melts_at_F(lld, Fr)
        ox = dict(ox)
        ox["H2O"] = H2O_melt[i]
        lg_melt = grd08_logeta(ox, T_C, Fmelt[i])

        phi = (1.0 - Fr) if phi_override is None else phi_override[i]
        locked = phi >= phi_max
        rx = eta_r_roscoe(phi, phi_max=phi_max, Bexp=Bexp)
        lg_bulk = float("inf") if locked else lg_melt + math.log10(rx)

        phi_b = phi_bub_post if sat[i] else 0.0
        rb = eta_r_bubble(phi_b, bubble_regime)
        lg_bub = float("inf") if locked else lg_bulk + math.log10(rb)

        out.append(dict(F=Fr, T_C=T_C, SiO2=ox["SiO2"], H2O=H2O_melt[i],
                        Fmelt=Fmelt[i], sat=sat[i], phi=phi, locked=locked,
                        lg_melt=lg_melt, lg_bulk=lg_bulk, lg_bub=lg_bub))
    return out


def print_table(rows):
    print("=" * 90)
    print("Bulk magma viscosity along the crystallisation path  eta(F)")
    print("log10(eta/Pa.s).  phi=1-F (Roscoe, phi_max=0.60).  '*'=H2O-saturated  "
          "'LOCK'=phi>=phi_max")
    print("=" * 90)
    hdr = (f"{'F':>5} {'T(C)':>6} {'SiO2':>6} {'H2O':>5} {'F_wt':>5} "
           f"{'phi':>5} | {'melt':>6} {'+xtl':>7} {'+bub':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sflag = "*" if r["sat"] else " "
        if r["locked"]:
            bulk, bub = " LOCK ", " LOCK "
        else:
            bulk, bub = f"{r['lg_bulk']:6.2f}", f"{r['lg_bub']:6.2f}"
        print(f"{r['F']:5.2f} {r['T_C']:6.0f} {r['SiO2']:6.1f} {r['H2O']:5.2f}{sflag}"
              f"{r['Fmelt']:5.2f} {r['phi']:5.2f} | {r['lg_melt']:6.2f}  {bulk}  {bub}")
    print("-" * len(hdr))


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["F", "T_C", "SiO2_norm", "H2O_melt_wtpct", "F_melt_wtpct",
                    "H2O_saturated", "phi_crystal", "locked",
                    "log_eta_melt", "log_eta_bulk_xtl", "log_eta_bulk_bubble"])
        for r in rows:
            w.writerow([f"{r['F']:.4f}", f"{r['T_C']:.1f}", f"{r['SiO2']:.2f}",
                        f"{r['H2O']:.3f}", f"{r['Fmelt']:.4f}", int(r["sat"]),
                        f"{r['phi']:.3f}", int(r["locked"]),
                        f"{r['lg_melt']:.3f}",
                        "" if r["locked"] else f"{r['lg_bulk']:.3f}",
                        "" if r["locked"] else f"{r['lg_bub']:.3f}"])
    print(f"[written] {path}")


def maybe_plot(rows, path, source=""):
    """Optional plot if matplotlib is available; otherwise skipped silently."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not available - skipped (csv still written).")
        return
    F = [r["F"] for r in rows]
    melt = [r["lg_melt"] for r in rows]
    bulk = [r["lg_bulk"] if not r["locked"] else float("nan") for r in rows]
    bub = [r["lg_bub"] if not r["locked"] else float("nan") for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(F, melt, "g-", lw=2, label="melt (GRD08)")
    ax.plot(F, bulk, "b-", lw=2, label="bulk (x crystals, phi=1-F)")
    ax.plot(F, bub, "b--", lw=1.5, label="bulk (+ bubbles)")
    sat_on = next((r["F"] for r in rows if r["sat"]), None)
    if sat_on is not None:
        ax.axvline(sat_on, color="c", ls=":", lw=1, label="H2O saturation")
    ax.axvline(0.5, color="r", ls=":", lw=1, label="phi=0.5 (mobile limit)")
    ax.invert_xaxis()  # crystallisation proceeds to the right
    ax.set_xlabel("Melt fraction F  (crystallisation ->)")
    ax.set_ylabel(r"$\log_{10}(\eta\,/\,\mathrm{Pa\,s})$")
    tag = source if source else "LLD"
    ax.set_title(f"Ok Tedi bulk magma viscosity vs melt fraction\n(LLD: {tag})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[written] {path}")


# =============================================================================
if __name__ == "__main__":
    # ---- Lormand volatile parameters (from CreateTargets defaults; swap in a
    #      multistart best-fit row when you have one) -------------------------
    LORMAND = {
        "H2O_init": 4.0, "Cl_init": 0.24, "F_init": 0.14,
        "D_xl_m_Cl": 0.2, "D_xl_m_F": 0.6, "D_xl_m_H2O": 0.12,
        "D_fl_m_Cl": 20.0, "D_fl_m_F": 4.0, "H2O_Sat": 8.5,
    }

    # ---- melt fraction grid (matches CreateTargets: 1 -> 0.1) ---------------
    N = 91
    F_grid = [1.0 - (1.0 - 0.1) * k / (N - 1) for k in range(N)]

    # ---- MELTS liquid line of descent: path from argv[1], else the example ---
    default_lld = os.path.join(HERE, "example_melts_lld.csv")
    lld_path = sys.argv[1] if len(sys.argv) > 1 else default_lld
    if not os.path.isabs(lld_path) and not os.path.exists(lld_path):
        lld_path = os.path.join(HERE, lld_path)
    lld = load_melts_lld(lld_path)
    stem = os.path.splitext(os.path.basename(lld_path))[0]
    print(f"[input] MELTS LLD: {os.path.basename(lld_path)}  "
          f"({len(lld['F'])} nodes, F {max(lld['F']):.2f} -> {min(lld['F']):.2f})")
    if "example" in os.path.basename(lld_path):
        print("[WARNING] using ILLUSTRATIVE example LLD - numbers are NOT Ok Tedi "
              "and NOT reportable [V6].\n")

    rows = viscosity_evolution(lld, LORMAND, F_grid,
                               phi_bub_post=0.10, bubble_regime="low_Ca")
    print_table(rows)
    write_csv(rows, os.path.join(HERE, f"{stem}__viscosity.csv"))
    maybe_plot(rows, os.path.join(HERE, f"{stem}__viscosity.png"),
               source=os.path.basename(lld_path))

    # quick scope-relevant readout: mobile window F in [0.5, 1.0]
    mob = [r for r in rows if r["F"] >= 0.5 and not r["locked"]]
    if mob:
        lo = min(r["lg_bulk"] for r in mob)
        hi = max(r["lg_bulk"] for r in mob)
        print(f"\nmobile window (F>=0.5, phi<=0.5): bulk log eta = "
              f"{lo:.2f} .. {hi:.2f}")
