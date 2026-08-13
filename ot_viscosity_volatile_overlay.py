#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ok Tedi P1 -- combined figure: melt-viscosity cooling path (LEFT axis, log eta)
FULLY COUPLED to the apatite-derived melt F-Cl-H2O trajectory (RIGHT axis, wt%).

Full coupling (per user):
  * H2O(T) and F(T) of the viscosity path are taken DIRECTLY from the apatite
    volatile model (`apatite_model.volatile_evolution`) of the OT-test-26.6.25
    inversion, mapped from melt fraction F onto temperature via the B2 cooling
    curve `OT-test-26.6.25_1000-750-B2.csv` (Var1=T_C, Var2=F).
  * T_sat (the viscosity kink) = the temperature at which melt H2O reaches
    saturation, i.e. B2(F_sat) -- NOT an assumed 750. So the kink, the volatile
    saturation onset, and the Cl peak all coincide by construction.
  * The Monte-Carlo band is propagated from the inversion ACCEPTED FAMILY
    (OT_inv_multistart.csv): each accepted parameter set -> its own volatile
    trajectory -> its own viscosity(T); the band is the percentile envelope. So
    the viscosity uncertainty is tied to the apatite non-uniqueness, not an
    ad-hoc triangular draw.

Comparison path: H2O frozen at the initial value (no enrichment) over the SAME
cooling temperatures and the SAME evolving melt F(T); isolates the water-
enrichment viscosity dividend.

Cl only matters for the story, not for viscosity (not a GRD08 input). Right axis
shows melt H2O and Cl x10, F x10 (magnified, annotated).

Majors for the viscosity path are the REAL OT MELTS LLD residual-melt majors
(24OT02-25 Kalgoorlie, `oktedi_melts_lld_25_*.csv`, 200 MPa baseline — LLD
melt-proxy ruling 2026-07-08, replaces 24OT02-57; anhydrous-renormalised,
interpolated per F via `melts_at_F`). The LLD is used as a COMPOSITION(F)
library only; the temperature axis stays the apatite-inversion B2 curve
(ruling: ONE T-F curve everywhere), so the LLD's own T(F) is deliberately NOT
used here. Besides the coupled overlay, this script also writes the MELT-ONLY
two-branch eta figure (same path, no volatile twin axis) = the manuscript's
main melt-only figure: ONE two-panel file `ot_viscosity_path_OTLLD_panels`, (a) melt
fraction LEFT + (b) temperature RIGHT, shared y, single legend outside right
(layout decision 2026-08-08; replaces the former single-axis `_vsT`/`_vsF`
pair). It carries constant-H2O reference isopleths (`ISO_H2O_WT`, 3-6 wt%;
same path, only H2O frozen); the counterfactual no-enrichment line is
DROPPED from that figure (decision 2026-08-08 -- its dividend stays in
key_numbers); the coupled overlay figure keeps the enriching + constant MC
bands and gains no isopleths.

Horizontal axis of the COUPLED overlay is switchable (`X_MODE` env: "both"
default / "t" / "f"): temperature (default) or melt fraction F
(re-parameterised T->F via inverse B2, everything else unchanged); those
files keep the `_vsT` / `_vsF` suffix. The melt-only panels figure is always
written once, regardless of X_MODE.

Prereq: run the canonical inversion first (inversion/ot_inversion_run.py —
writes OT_inv_median_params.csv + OT_inv_multistart.csv into the run dir).
Env: OT_DATA_DIR (run dir), OT_LLD, X_MODE.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import apatite_model as am          # noqa: E402
import viscosity_path as vp         # noqa: E402
from magma_viscosity_evolution import load_melts_lld, melts_at_F  # noqa: E402

LLD_PATH = Path(os.environ.get("OT_LLD") or (_SCRIPT_DIR / "oktedi_melts_lld_25_260708.csv"))

# Canonical data dir = a COMPOSED inversion run dir: central = the reported
# median params, family = the converged multistart, band anchor file alongside.
# It lives outside this repo (manuscript supplement); point OT_DATA_DIR at it.
_data_dir = os.environ.get("OT_DATA_DIR")
if not _data_dir:
    sys.exit("OT_DATA_DIR not set: point it at the composed inversion run dir "
             "(needs OT_inv_median_params.csv, OT_inv_multistart.csv and the "
             "B2 T-F curve OT-test-26.6.25_1000-750-B2.csv)")
HERE = Path(_data_dir)
TF_PATH = HERE / "OT-test-26.6.25_1000-750-B2.csv"
# manuscript central = accepted-family MEDIAN (ruling 2026-07-07) — the single
# best point is degenerate along H2O_init/H2O_Sat under logRMSE.
BESTFIT = HERE / "OT_inv_median_params.csv"
FAMILY = HERE / "OT_inv_multistart.csv"

NUM_F = 200                          # melt-fraction grid for the volatile model
RMSE_MULT = 5.0                      # accepted family = RMSE <= RMSE_MULT x best
PARAM_KEYS = ["H2O_init", "Cl_init", "F_init", "D_xl_m_Cl", "D_xl_m_F",
              "D_xl_m_H2O", "D_fl_m_Cl", "D_fl_m_F", "H2O_Sat"]

# H2O_Sat is only WEAKLY constrained by apatite Cl-OH-F (the accepted family spreads
# it across the whole 8-11 bound; corr(RMSE, H2O_Sat) ~= 0.07), so the plotted
# trajectory + family band PIN it to ONE declared value and keep the genuine H2O
# uncertainty in H2O_init. Declaration unified 2026-07-08: ALL figures from this
# pipeline state the AS-INVERTED family-median H2O_Sat (the declared canonical
# value); figure captions must say so.
# Ruling 2026-07-07 (5) unchanged: the depth / H2O_Sat control stays OPEN (a
# follow-up study) — this pin is a stated display choice, not a resolved
# constraint.
# "median" -> pin at the median-params file's own H2O_Sat (resolved in main());
# a float pins at that value (10.0 reproduces the retired pin-10 P3-only variant);
# None -> each accepted fit keeps its own H2O_Sat.
H2O_SAT_FIX = "median"

# Horizontal-axis mode: "both" (default) writes both figures, "t" temperature only,
# "f" melt-fraction only. Override via the X_MODE env var.
X_MODE = os.environ.get("X_MODE", "both").lower()

# RULING 2026-07-08: rhyolite-MELTS stays the majors library over the FULL F range,
# but below this F the engines diverge (MAGEMin crystallises biotite for 24OT02-25
# from F 0.61-0.76; the MELTS<->MAGEMin cross-check is quoted for F>=0.7 only) —
# shade that zone in the figures and state it in captions.
F_ENGINE_DEP = 0.63

# Constant-H2O reference isopleths on the MELT-ONLY P1 figure (added 2026-08-08):
# same path (LLD majors(F), halogen F(T), B2 T axis), only H2O frozen. 3-6 wt%
# brackets the family H2O_init envelope (~2.6-5.5) and stays inside the GRD08
# ~8 wt% calibration. On that figure they replace the counterfactual's MC band
# (median line kept); the coupled overlay figure is unchanged.
ISO_H2O_WT = [3.0, 4.0, 5.0, 6.0]

# FIXED manuscript frame for the melt-only two-panel P1 figure (user decision
# 2026-08-08): log10 eta 2-5 shared, (a) melt fraction 1.0->0.0, (b) temperature
# 1000->750 C (= the full B2 path range). Fixed rather than autoscaled so the
# panels stay directly comparable across re-runs and against the other eta
# figures; both x ranges are given in display order (left, right). The COUPLED
# overlay figures keep autoscale.
# y range first set 2-8, narrowed to 2-5 the same day: everything drawn spans
# log10 eta 2.05-4.17 (MC95 up to 3.40; the 3 wt% isopleth is the ceiling at
# 4.17), so 2-5 still clips nothing while roughly doubling the vertical scale
# and restoring the T_sat kink that 2-8 flattened.
# y frame lowered 2-5 -> 1.5-5 on 2026-08-10 with the canonical replacement:
# the new central dips to log eta 1.96 at T_sat and the adopted band's lower
# edge reaches 1.81 (measured drawn extent 1.81-4.17), so the 2026-08-08 frame
# clips. 1.5-5 clears it with margin; the retired 26.7.7 variant rendered at
# (2, 5) (archived figures unchanged; pass OT_PANEL_YLIM=2,5 to reproduce).
# ⚠ apatite_gui.PREVIEW_YLIM duplicates this ON PURPOSE — change both together.
PANEL_YLIM = (1.5, 5.0)
PANEL_XLIM_F = (1.0, 0.0)
PANEL_XLIM_T = (1000.0, 750.0)
# OT_PANEL_YLIM ("lo,hi") overrides the frame for what-if renders — any
# override must be flagged in the run report; OT_RUN_TAG names the inversion
# run the figure notes cite (default 26.8.10 = the standing canonical).
_env_ylim = os.environ.get("OT_PANEL_YLIM")
if _env_ylim:
    PANEL_YLIM = tuple(float(v) for v in _env_ylim.split(",")[:2])
RUN_TAG = os.environ.get("OT_RUN_TAG", "26.8.10")

# Manuscript-panels BAND definition (user rulings 2026-08-09, THIS figure only;
# the coupled overlay keeps the 5x-family percentile band and the H2O_Sat pin).
#
# WHY IT IS A WINDOW AROUND THE CENTRAL SOLUTION, not "<= t x best".
# Re-reading OT_inv_multistart.csv (2026-08-09, third round) established:
#   * the accepted family is a near-perfect 1-D RIDGE -- Spearman rho = +0.999
#     to +1.000 between H2O_init, H2O_Sat, Cl_init and F_init -- so a solution
#     is fixed by ONE ridge coordinate (take H2O_init), and misfit rises
#     MONOTONICALLY along it from the low-H2O end;
#   * the canonical central solution (median params, ruling 2026-07-07) IS a
#     family member: rank 187/400, misfit 1.099x best, reproducing the central
#     curve to max|diff| = 0.000 log.
# Therefore ANY one-sided cut "RMSE <= t x best" starts at the low-H2O/best
# end and can only reach the centre by growing until t > 1.10; even then the
# centre rides the band edge (position in band: 0.07 at t=1.11, 0.29 at 1.15,
# 0.43 at 1.20). That is the entire cause of the poor centre/band overlap --
# an incompatible selection rule, not a modelling error.
# So membership is a SYMMETRIC MISFIT WINDOW around the central solution:
#     |RMSE - RMSE_central| <= BAND_MISFIT_FRAC * RMSE_central
# which contains the centre by construction (measured position 0.45 = mid-band)
# and still gives a narrow band: ~81 members, 0.35 log wide -- the size and
# width the "+/-20%" request was after.
# Each member keeps its OWN as-inverted H2O_Sat (band un-pinned; the central
# line still runs on the declared family median = its own as-inverted value),
# so the saturated branch retains real width.
# SHADING (user ruling 2026-08-09, later): members are sorted by INITIAL MELT
# H2O and the band is filled between consecutive members, light = low H2O ->
# dark = high H2O, with the colorbar reading directly in wt%. This replaces the
# earlier |misfit offset| gradient (nested envelopes, dark at the centre).
# Membership is untouched -- see the paragraph above; the colour key is a
# presentation choice, the selection rule is not.
# NOTE selection MUST be misfit-based, never parameter-based: the multistart
# file contains diverged runs (max RMSE ~6.7e8 x best) that a parameter-space
# box would silently admit.
BAND_MISFIT_FRAC = float(os.environ.get("OT_BAND_FRAC", "0.03"))
# ^ env override (default 0.03 = the ruled window) exists for the 2026-08-10
# band re-ruling exhibits: OT_BAND_FRAC=999 with a pre-trimmed FAMILY csv
# renders "membership = the whole file" (e.g. a top-N subset as the band).
BAND_CMAP = "Blues"
BAND_CMAP_DEPTHS = (0.15, 0.85)          # colormap sample: low-H2O, high-H2O end
# The lowest-misfit path was drawn alongside the median as an orange second
# reference line (2026-08-09 rounds 2-3). DROPPED by user ruling the same day:
# it sat +0.52 log ABOVE the band -- the mirror of the problem the band
# redefinition had just fixed -- and the ridge/end-member argument it was there
# to illustrate is carried by the figure note and the caption instead.


def load_B2():
    """Return (T_of_F, F_of_T) monotone interpolators from the B2 cooling curve."""
    if not TF_PATH.exists():
        sys.exit(f"B2 curve not found: {TF_PATH}")
    tf = np.genfromtxt(TF_PATH, delimiter=",", names=True)
    T = np.asarray(tf["Var1"], float)
    F = np.asarray(tf["Var2"], float)
    oF = np.argsort(F)
    Fs, Ts = F[oF], T[oF]
    oT = np.argsort(T)
    Ta, Fa = T[oT], F[oT]

    def T_of_F(fvals):
        return np.interp(np.asarray(fvals, float), Fs, Ts)

    def F_of_T(tvals):
        return np.interp(np.asarray(tvals, float), Ta, Fa)
    return T_of_F, F_of_T


def params_dict(row, pin_h2o_sat=True):
    """pin_h2o_sat=False keeps the row's own as-inverted H2O_Sat — used ONLY
    for the manuscript panels band (user 2026-08-09 ruling: that figure's band
    members un-pin; the central path, the coupled overlay and every other
    figure keep the unified declared H2O_Sat)."""
    p = {k: float(row[k]) for k in PARAM_KEYS}
    p["D_fl_m_PrimBoilH2O"] = 0.0
    if pin_h2o_sat and H2O_SAT_FIX is not None:       # pin the weakly-constrained H2O_Sat
        p["H2O_Sat"] = H2O_SAT_FIX
    return p


def volatiles_on_T(p, F, T_of_F):
    """Forward the volatile model for params p; return melt wt% arrays and the
    saturation temperature, all placed on the temperature axis via B2."""
    vol = am.volatile_evolution(F, p)
    H2O = vol["H2O_rem"] * 100.0
    Cl = vol["Cl_rem"] * 100.0
    Ff = vol["F_rem"] * 100.0
    Tv = T_of_F(F)                                   # 1000 -> ~829 as F 1 -> 0.1
    sat_i = int(np.argmax(vol["Fluid_inc_mass"] > 0)) if (vol["Fluid_inc_mass"] > 0).any() else None
    T_sat = float(Tv[sat_i]) if sat_i is not None else None
    F_sat = float(F[sat_i]) if sat_i is not None else None
    return dict(T=Tv, F_melt=F, H2O=H2O, Cl=Cl, F=Ff, T_sat=T_sat, F_sat=F_sat)


def coupled_logeta(T_grid, majors_grid, av, H2O_sat, freeze_H2O=None):
    """log eta along T_grid using H2O(T) & F(T) ingested from an apatite volatile
    trajectory `av` (dict T/H2O/F, T decreasing). `majors_grid` = per-T_grid list
    of major-oxide dicts (real OT LLD, composition(F) via inverse-B2 F(T)).
    Above T_sat use the ingested (rising) H2O; below, hold H2O at H2O_sat.
    `freeze_H2O` (e.g. the initial value) overrides H2O everywhere for the
    no-enrichment comparison path."""
    order = np.argsort(av["T"])
    Ts, H2Os, Fs = av["T"][order], av["H2O"][order], av["F"][order]
    Tmin = Ts[0]
    T_sat = av["T_sat"] if av["T_sat"] is not None else Tmin
    le = np.empty(len(T_grid))
    for i, T in enumerate(T_grid):
        Fm = float(np.interp(T, Ts, Fs))             # evolving melt F (frozen below range)
        if freeze_H2O is not None:
            h = freeze_H2O
        elif T >= T_sat:
            h = float(np.interp(T, Ts, H2Os))        # ingested rising H2O
        else:
            h = H2O_sat                              # buffered constant below sat
        le[i] = vp.log_eta({**majors_grid[i], "H2O": h, "F": Fm}, T)
    return le


def main():
    vp.validate() or sys.exit("Giordano model failed Table 2 validation -- refusing.")
    if not BESTFIT.exists() or not FAMILY.exists():
        sys.exit(f"inversion outputs missing under {HERE}\n"
                 "  run inversion/ot_inversion_run.py first.")

    T_of_F, F_of_T = load_B2()
    F = np.linspace(1.0, 0.1, NUM_F)

    bf = pd.read_csv(BESTFIT).iloc[0]
    global H2O_SAT_FIX
    if isinstance(H2O_SAT_FIX, str):        # "median": pin at the as-inverted value
        H2O_SAT_FIX = float(bf["H2O_Sat"])
        print(f"H2O_Sat pinned at the as-inverted central value: {H2O_SAT_FIX:.2f} wt% "
              "(declaration unified across P1 figures, 2026-07-08)")
    p_best = params_dict(bf)
    av = volatiles_on_T(p_best, F, T_of_F)
    H2O_sat = p_best["H2O_Sat"]
    print(f"=== family-median volatile trajectory (H2O_Sat={H2O_sat:.2f}) ===")
    print(f"  melt wt% start H2O/Cl/F: {av['H2O'][0]:.2f}/{av['Cl'][0]:.3f}/{av['F'][0]:.3f}")
    print(f"  melt wt% end   H2O/Cl/F: {av['H2O'][-1]:.2f}/{av['Cl'][-1]:.3f}/{av['F'][-1]:.3f}")
    print(f"  Cl peak {av['Cl'].max():.3f} wt% ; H2O-sat onset T_sat = {av['T_sat']:.0f} C")

    # ---- viscosity T grid (within the modelled 1000-750 window) ----
    T_INITIAL, T_FINAL = vp.T_INITIAL, 750.0
    n = int(round((T_INITIAL - T_FINAL) / vp.DT)) + 1
    T_grid = np.linspace(T_INITIAL, T_FINAL, n)
    # REAL OT majors: LLD composition(F) placed on the B2 temperature axis
    # (one majors dict per T_grid node; same for every family member).
    lld = load_melts_lld(str(LLD_PATH))
    majors_grid = [melts_at_F(lld, float(f))[1] for f in F_of_T(T_grid)]
    print(f"  majors: real OT MELTS LLD ({LLD_PATH.name}), composition(F) on the B2 T axis")

    # deterministic coupled path + no-enrichment comparison
    le_path = coupled_logeta(T_grid, majors_grid, av, H2O_sat)
    le_cf = coupled_logeta(T_grid, majors_grid, av, H2O_sat, freeze_H2O=p_best["H2O_init"])
    # constant-H2O reference isopleths (deterministic; family-median F(T) trajectory)
    iso = dict(values=ISO_H2O_WT,
               curves=[coupled_logeta(T_grid, majors_grid, av, H2O_sat, freeze_H2O=w)
                       for w in ISO_H2O_WT])
    T_sat = av["T_sat"]
    branch = np.array(["under" if T >= T_sat else "sat" for T in T_grid], dtype=object)

    # ---- MC band from the inversion accepted family ----
    fam = pd.read_csv(FAMILY)
    fam.columns = PARAM_KEYS + ["RMSE"]               # rename to engine keys
    best_rmse = fam["RMSE"].min()
    acc = fam[fam["RMSE"] <= RMSE_MULT * best_rmse].reset_index(drop=True)
    print(f"  accepted family for band: {len(acc)}/{len(fam)} (RMSE <= {RMSE_MULT}x best)")
    le_fam = np.empty((len(acc), len(T_grid)))
    le_fam_cf = np.empty((len(acc), len(T_grid)))
    for j in range(len(acc)):
        pj = params_dict(acc.iloc[j])
        avj = volatiles_on_T(pj, F, T_of_F)
        le_fam[j] = coupled_logeta(T_grid, majors_grid, avj, pj["H2O_Sat"])
        le_fam_cf[j] = coupled_logeta(T_grid, majors_grid, avj, pj["H2O_Sat"],
                                      freeze_H2O=pj["H2O_init"])

    def pct(mat):
        return dict(median=np.median(mat, axis=0), p2_5=np.percentile(mat, 2.5, axis=0),
                    p16=np.percentile(mat, 16, axis=0), p84=np.percentile(mat, 84, axis=0),
                    p97_5=np.percentile(mat, 97.5, axis=0))
    pct_path, pct_const = pct(le_fam), pct(le_fam_cf)

    # ---- manuscript-panels band: misfit window AROUND THE CENTRE (2026-08-09,
    # third round -- see the BAND_MISFIT_FRAC block for the diagnosis) ----
    from matplotlib import pyplot as plt_cm     # colormap sampling only
    # locate the central (median-params) solution IN the family by parameter
    # match, so the window is anchored on the curve actually drawn
    # The window ANCHOR defaults to the drawn central (bf). OT_BAND_ANCHOR
    # (path to a params csv in the median-file schema) decouples them — needed
    # since the 2026-08-09-night ruling put the central on one family
    # (ftol=0.01 top-5% median) while the band is measured on another (the
    # converged 1e-10 family): anchoring that window on the 0.01 central's own
    # misfit (a different scale) would catch nobody.
    _anchor_env = os.environ.get("OT_BAND_ANCHOR")
    # canonical-dir convention (2026-08-10): if OT_inv_band_anchor_params.csv
    # sits next to the median file, it IS the window anchor (the converged
    # family's best-region params) — no env choreography needed for the
    # canonical render; the env still wins when set.
    _anchor_auto = HERE / "OT_inv_band_anchor_params.csv"
    if not _anchor_env and _anchor_auto.exists():
        _anchor_env = str(_anchor_auto)
    anchor = pd.read_csv(_anchor_env).iloc[0] if _anchor_env else bf
    if _anchor_env:
        print(f"  [band] window anchor decoupled from the drawn central: "
              f"{Path(_anchor_env).name}")
    rel = np.zeros(len(fam))
    for k in PARAM_KEYS:
        scale = max(abs(float(anchor[k])), 1e-9)
        rel += np.abs(fam[k].to_numpy() - float(anchor[k])) / scale
    j_cen = int(np.argmin(rel))
    rmse_cen = float(fam.iloc[j_cen]["RMSE"])
    if rel[j_cen] > 1e-3:                       # not a family member -> fall back
        # The componentwise median need not BE a family member (26.7.7's 1-D
        # ridge made it one; a loose-ftol fan does not). The definition wants
        # the window anchored on the CENTRAL SOLUTION'S OWN misfit, so prefer
        # the RMSE_central column the inversion wrapper computes by forwarding
        # the median params on the fit grid (current inversion runner); only if
        # that is absent anchor on the nearest member and say so.
        if "RMSE_central" in anchor.index and np.isfinite(float(anchor["RMSE_central"])):
            rmse_cen = float(anchor["RMSE_central"])
            print(f"  [band] anchor params are not a family member (closest "
                  f"rel.dev {rel[j_cen]:.3g}); window anchored on the anchor "
                  f"curve's OWN misfit RMSE_central = {rmse_cen:.6g}")
        else:
            print(f"  [band] WARNING central params not found in the family "
                  f"(closest rel.dev {rel[j_cen]:.3g}); anchoring on it anyway")
    off = np.abs(fam["RMSE"].to_numpy() - rmse_cen) / rmse_cen   # |misfit offset|
    sel = off <= BAND_MISFIT_FRAC
    top = fam[sel].reset_index(drop=True)
    n_top = len(top)
    print(f"  panels band: |RMSE - RMSE_central| <= {BAND_MISFIT_FRAC:.0%} of "
          f"RMSE_central -> {n_top}/{len(fam)} members "
          f"(central solution rank {int((fam['RMSE'] < rmse_cen).sum()) + 1}, "
          f"misfit {rmse_cen / best_rmse:.3f}x best; window spans "
          f"{fam['RMSE'][sel].min() / best_rmse:.3f}-"
          f"{fam['RMSE'][sel].max() / best_rmse:.3f}x); own H2O_Sat "
          f"{top['H2O_Sat'].min():.2f}-{top['H2O_Sat'].max():.2f} wt% (UN-pinned)")
    le_top = np.empty((n_top, len(T_grid)))
    for j in range(n_top):
        pj = params_dict(top.iloc[j], pin_h2o_sat=False)
        avj = volatiles_on_T(pj, F, T_of_F)
        le_top[j] = coupled_logeta(T_grid, majors_grid, avj, pj["H2O_Sat"])
    cmap = plt_cm.get_cmap(BAND_CMAP)
    if n_top < 2:
        # DEGENERATE-BAND GUARD (first hit 2026-08-09 by the ms2000 ftol=0.01
        # candidate): in a loose-ftol fan the member misfits are dominated by
        # unconverged-optimizer noise, not physical position (rho(H2O_init,
        # RMSE) ~ +0.05 there vs monotone on the converged 26.7.7 ridge), so
        # the +/-3% window can catch <2 members. An EMPTY band_layers list
        # deliberately draws nothing -- _draw_panel then also suppresses the
        # legacy pct fills, rather than silently reverting to the percentile
        # band the 2026-08-09 rulings replaced. Such a family needs a user
        # re-ruling on band membership (run a converged ftol=1e-10 reference).
        print(f"  [band] DEGENERATE: only {n_top} member(s) inside the "
              f"+/-{BAND_MISFIT_FRAC:.0%} misfit window -> rendering the "
              "central line WITHOUT a band or colorbar; membership needs a "
              "re-ruling for this family (see run log).")
        band_layers, band_cbar = [], None
        h_lo = h_hi = float(bf["H2O_init"])
        panels_note = (
            "MELT viscosity only (no crystal/bubble loading); isobaric; majors = "
            "real OT MELTS LLD composition(F), 24OT02-25. Central path (blue): "
            f"family MEDIAN, declared H2O_Sat = {H2O_sat:.2f} wt% (as-inverted "
            f"{RUN_TAG} median). Band OMITTED: only {n_top} member(s) fall "
            f"within {BAND_MISFIT_FRAC:.0%} of the central solution's misfit "
            "-- the misfit-window band presupposes a converged family, and "
            "this run's loose-ftol members carry optimizer noise in their "
            "misfits; band membership awaits a re-ruling.")
    else:
        # ---- SHADE THE BAND BY INITIAL MELT H2O (user ruling 2026-08-09) ----
        # MEMBERSHIP IS UNCHANGED and stays misfit-based (see the
        # BAND_MISFIT_FRAC block: a parameter-space box would silently admit
        # the diverged runs). Only the colour key moves, from |misfit offset|
        # to H2O_init -- the quantity the band is actually about, and the one
        # the reader wants ordered.
        #
        # Why per-member fills tile cleanly (measured 2026-08-09, 84 members):
        #   * eta is monotone in H2O_init at EVERY T node -- Spearman rho =
        #     -0.9998 to -1.0000 -- because the window sits on the 1-D
        #     H2O_init-H2O_Sat ridge (rho = +0.9998 within the band);
        #   * the top edge of the band is the lowest-H2O member and the bottom
        #     edge the highest, at every node, so filling between consecutive
        #     members sorted by H2O_init reproduces the band extent EXACTLY,
        #     with no gaps;
        #   * only 2.9% of consecutive pairs invert, by at most 0.007 log
        #     against a 0.35 log band -- below the line width.
        # Colour is therefore linear in H2O_init, and the colorbar bounds are
        # the members' own H2O_init values, so the bar reads directly in wt%.
        # NOTE the monotonicity above was MEASURED on the 26.7.7 ridge family;
        # re-measure it whenever the band is rebuilt on a new multistart.
        h2o_init = top["H2O_init"].to_numpy()
        order = np.argsort(h2o_init)
        le_ord, h2o_ord = le_top[order], h2o_init[order]
        h_lo, h_hi = float(h2o_ord[0]), float(h2o_ord[-1])
        d_lo, d_hi = BAND_CMAP_DEPTHS                  # low-H2O end, high-H2O end
        def _shade(h):                                 # linear in wt%
            t = (h - h_lo) / (h_hi - h_lo) if h_hi > h_lo else 0.5
            return cmap(d_lo + (d_hi - d_lo) * t)
        # The legend swatch goes on a MIDDLE layer, not the first: layer 0 is
        # the low-H2O end and is nearly white, which reads as an empty box in
        # the legend. The colorbar already names the colour key, so the legend
        # entry only has to state MEMBERSHIP.
        k_leg = (n_top - 1) // 2
        band_layers, band_colors = [], []
        for k in range(n_top - 1):
            col = _shade(0.5 * (h2o_ord[k] + h2o_ord[k + 1]))
            band_colors.append(col)
            band_layers.append(dict(lo=np.minimum(le_ord[k], le_ord[k + 1]),
                                    hi=np.maximum(le_ord[k], le_ord[k + 1]),
                                    color=col,
                                    label=(f"misfit window around the central "
                                           f"solution (+/-{BAND_MISFIT_FRAC:.0%})"
                                           if k == k_leg else "_nolegend_")))
        print(f"  band shading: initial melt H2O {h_lo:.2f}-{h_hi:.2f} wt% "
              f"(central {float(bf['H2O_init']):.2f}), light = low -> dark = high; "
              f"{n_top - 1} per-member fills")
    # frame check (print-only): everything the PANELS figure draws vs the fixed
    # frame — the rounds of 2026-08-08/09 measured this by hand every time.
    _drawn = np.vstack([le_path[None, :], le_top]
                       + [np.asarray(c)[None, :] for c in iso["curves"]])
    print(f"  panels drawn extent: log10 eta {_drawn.min():.2f}-{_drawn.max():.2f} "
          f"vs fixed frame {PANEL_YLIM}"
          + ("  [CLIPPED — widen via OT_PANEL_YLIM and flag it]"
             if _drawn.min() < PANEL_YLIM[0] or _drawn.max() > PANEL_YLIM[1] else ""))
    if n_top >= 2:
        # BoundaryNorm needs STRICTLY increasing bounds; ties in H2O_init are
        # not present in the 26.7.7 family (min gap 2.4e-4 wt%) but would crash
        # a re-run on a coarser multistart, so nudge any that appear.
        bnds = np.asarray(h2o_ord, float).copy()
        eps = max((h_hi - h_lo) * 1e-6, 1e-9)
        for k in range(1, len(bnds)):
            if bnds[k] <= bnds[k - 1]:
                bnds[k] = bnds[k - 1] + eps
        # colorbar ticks are DATA-DRIVEN: the band's H2O_init range is
        # run-dependent (hardcoded [3.3, 3.5, 3.7, 3.9] for the 26.7.7 family
        # until 2026-08-09; a 26.7.7 re-render now shows 3.4/3.6/3.8 —
        # cosmetic only)
        _span = h_hi - h_lo
        _step = 0.1 if _span <= 0.45 else (0.2 if _span <= 0.9 else 0.5)
        _t0 = np.ceil(h_lo / _step) * _step
        cb_ticks = [round(float(t), 2) for t in np.arange(_t0, h_hi + 1e-9, _step)]
        band_cbar = dict(colors=band_colors,
                         bounds=[float(b) for b in bnds],
                         ticks=cb_ticks,
                         label="initial melt H2O (wt%)")
        # Note wording follows the 2026-08-10 canon by default; OT_NOTE_LEGACY
        # restores the 26.7.7-era sentences for archival re-renders.
        _legacy = bool(os.environ.get("OT_NOTE_LEGACY"))
        _central_desc = ("family MEDIAN" if _legacy else
                         "median of the best-5% starts (ftol=0.01 multistart, "
                         "ruling 2026-08-09)")
        _why = (("The median is reported rather than the best fit because "
                 "misfit rises monotonically along the 1-D H2O_init-H2O_Sat "
                 "degeneracy ridge, making the lowest-misfit solution a low-H2O "
                 "end-member rather than a central estimate.") if _legacy else
                ("The fully-converged best fit is not used as a point estimate: "
                 "it runs onto the prior-box faces (D_fl_m_Cl, then D_fl_m_F, "
                 "then the H2O_Sat floor) — a constrained end-point, not a "
                 "central tendency. Band membership is measured on the "
                 "converged (ftol=1e-10) family around its best-fit region; "
                 "the band's H2O_Sat lower edge is truncated by the 7.71 wt% "
                 "prior floor."))
        panels_note = ("MELT viscosity only (no crystal/bubble loading); isobaric; majors = "
                       "real OT MELTS LLD composition(F), 24OT02-25. Central path (blue): "
                       f"{_central_desc}, declared H2O_Sat = {H2O_sat:.2f} wt% (as-inverted "
                       f"{RUN_TAG} central). Band: the "
                       f"{n_top} solutions whose misfit is within "
                       f"{BAND_MISFIT_FRAC:.0%} of the window anchor's "
                       f"({rmse_cen / best_rmse:.2f}x the lowest), each retaining its own "
                       f"as-inverted H2O_Sat ({top['H2O_Sat'].min():.1f}-"
                       f"{top['H2O_Sat'].max():.1f} wt%), shaded by INITIAL melt H2O "
                       f"({h_lo:.2f}-{h_hi:.2f} wt%, light = low -> dark = high; colorbar). "
                       + _why)

    # ---- assemble dicts for vp.make_figure ----
    path = dict(T_C=T_grid, H2O_wt=np.array([np.interp(T, np.sort(av["T"]),
                av["H2O"][np.argsort(av["T"])]) if T >= T_sat else H2O_sat for T in T_grid]),
                logEta_path=le_path, logEta_cf=le_cf, branch=branch, coupling="apatite-coupled",
                T_sat=T_sat,
                majors_note="real OT MELTS LLD majors (24OT02-25), composition(F) on the B2 T axis",
                majors_tag="OT LLD majors (24OT02-25)",
                caveat=("majors = real OT MELTS LLD residual-melt composition(F) "
                        "(24OT02-25, 200 MPa baseline); MELT viscosity only "
                        "(no crystal/bubble loading); isobaric (no degassing)."),
                fig_note=("MELT viscosity only (no crystal/bubble loading); isobaric; majors = "
                          "real OT MELTS LLD composition(F), 24OT02-25. "
                          f"H2O_Sat = {H2O_sat:.2f} wt% (as-inverted {RUN_TAG} central) — "
                          "near the upper ~8 wt% GRD08 calibration limit."))
    slopes = vp.branch_slopes(path)
    volatiles = dict(T=av["T"], H2O=av["H2O"], Cl=av["Cl"], F=av["F"],
                     cl_scale=10.0, f_scale=10.0,
                     source=f"apatite canonical inversion family median ({HERE.name}), F->T via B2 (T_sat={T_sat:.0f}C). "
                            f"Saturated branch H2O={H2O_sat:.2f} wt% EXCEEDS the GRD08 ~8 wt% "
                            f"calibration (extrapolated)")

    # ---- switchable horizontal axis: temperature (default) or melt fraction F ----
    # T-axis = None (make_figure default); F-axis re-parameterises T->F via inverse B2,
    # with everything else unchanged. X_MODE selects which figure(s) to write.
    F_sat = av["F_sat"]
    xaxis_F = dict(path=F_of_T(T_grid), vol=av["F_melt"], sat=F_sat,
                   label="melt fraction F (crystallisation →)",
                   sat_label=f"F_sat={F_sat:.2f}", invert=True)
    # engine-dependent majors zone (F < F_ENGINE_DEP), expressed on each axis
    zlabel = f"majors engine-dependent (F<{F_ENGINE_DEP:g})"
    zone_T = dict(x0=float(T_of_F(F_ENGINE_DEP)), x1=float(T_grid.min()), label=zlabel)
    zone_F = dict(x0=F_ENGINE_DEP, x1=float(np.min(xaxis_F["path"])), label=zlabel)
    modes = {"T": (None, "vsT", zone_T), "F": (xaxis_F, "vsF", zone_F)}
    sel = {"t": ["T"], "f": ["F"], "both": ["T", "F"]}.get(X_MODE, ["T", "F"])

    print("\n" + vp.key_numbers(path, slopes, T_grid, pct_path))
    for m in sel:
        xa, sfx, zn = modes[m]
        # coupled overlay (viscosity + volatile twin axis)
        stem = f"ot_viscosity_volatiles_coupled_OTLLD_{sfx}"
        pdf, png = vp.make_figure(path, T_grid, pct_path, pct_const, stem,
                                  volatiles=volatiles, xaxis=xa, zone=zn)
        print(f"wrote ({m}-axis): {png}")
    # MELT-ONLY two-branch eta — the manuscript's main melt-only figure: ONE two-panel
    # file, (a) melt fraction LEFT + (b) temperature RIGHT, shared y, single
    # legend outside right (user decisions 2026-08-08; replaces the
    # single-axis vs{T,F} pair). Constant-H2O isopleths 3-6 wt% carry the
    # reference; the no-enrichment comparison line itself is dropped from the
    # figure (its dividend stays in key_numbers above); enriching MC band
    # kept; zone shading per the T2 "every eta figure" ruling (2026-07-08).
    pdf, png, svg = vp.make_figure_panels(
        path, T_grid, pct_path, pct_const, "ot_viscosity_path_OTLLD_panels",
        panels=[dict(xaxis=xaxis_F, zone=zone_F, letter="(a)", xlim=PANEL_XLIM_F),
                dict(xaxis=None, zone=zone_T, letter="(b)", xlim=PANEL_XLIM_T)],
        iso_h2o=iso, const_line=False, ylim=PANEL_YLIM,
        band_layers=band_layers, band_cbar=band_cbar, fig_note=panels_note)
    print(f"wrote (melt-only P1, 2-panel F|T): {png}")
    print(f"wrote (melt-only P1, vector): {svg}")
    print(f"\nFULL COUPLING from apatite inversion + B2. T_sat={T_sat:.0f}C / F_sat={F_sat:.2f}. "
          f"Majors = real OT MELTS LLD (24OT02-25). H2O_Sat as-inverted "
          f"{H2O_sat:.2f} wt%. X_MODE={X_MODE}.")


if __name__ == "__main__":
    main()
