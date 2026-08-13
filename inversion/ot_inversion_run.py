#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ok Tedi apatite Cl-OH-F inversion — OT-test-26.8.9_ms2000 (**CANDIDATE** run;
canonical decision PENDING — user ruling 2026-08-09: "先出数字再定", so this
run replaces nothing until the old-vs-new comparison is reviewed).

WHAT THIS IS: a headless, GUI-EQUIVALENT re-run of the OT-APA-TEST-3.otproj
manifest state (saved 2026-08-09 02:15 = the project's run_014 configuration,
first executed inside the GUI 2026-08-07 01:17). The user revised the hand
eye-fit target in the GUI; this wrapper reproduces that exact inversion at
n_start=2000 and writes the canonical OT_inv_* file set for the downstream
viscosity overlay.

Differences from the 26.7.7 canonical run (ALL user-set in the project):
  * TARGET params revised (e.g. D_xl_m_Cl 0.05 -> 0.2103, D_fl_m_Cl 22 ->
    25.95, H2O_Sat target 10 -> 11.02, F_init 0.46 -> 0.354);
  * K_SPEC 0.463 -> 0.388;
  * BOUNDS revised (e.g. D_xl_m_Cl 0.1472-0.2734, H2O_Sat 7.7129-12.0);
  * n_start 400 -> 2000;
  * ms_ftol = 0.01 (loose, Lormand-style早停) — user ruling 2026-08-09:
    run AS CONFIGURED in the project. ⚠ engine docstring: a loose-ftol
    family "fans out over the data — DISPLAY/EXPLORATION only, not honest
    uncertainty"; if this run becomes canonical the manuscript must carry
    that caveat (the 26.7.7 canonical family was fully converged, 1e-10).

GUI EQUIVALENCE (why this does NOT call am.build_F_and_T directly):
the GUI's grids come from a DOUBLE interpolation — apatite_gui.load_tf_curve
first regrids the raw B2 csv to NUM_F_LIVE=120 points via am.build_F_and_T,
then _grid(n) re-interpolates that 120-pt polyline to n points (target n=200,
fit n=NUM_F_FIT=100). am.build_F_and_T(raw, n) directly would differ in the
last few decimals. This wrapper imports apatite_gui and reuses its
load_tf_curve / compute_state / NUM_F_FIT so target and objective are
bit-identical to what the GUI ran; start points are drawn exactly as
gui._ms_worker does (default_rng(inv.SEED), one 9-vector per start).

GOLDEN: the same config already ran inside the GUI as run_014 (n=2000,
best_rmse 0.00864008234041, recorded in the .otproj manifest). Reproducing
that value here validates the headless pathway end-to-end. (scipy version
differences would surface as last-digit drift — report, don't hide, if the
match is not exact.)

FAMILY MEDIAN acceptance: RMSE <= 5 x best (user ruling 2026-08-09 —
canonical continuity with 26.7.7; the project's rmse_mult=10 GUI display
setting is deliberately NOT used for the manuscript-central median).

Env: OT_DATA_DIR (run dir; REQUIRED — holds the observed-apatite CSV + the
B2 T-F curve), NUM_START, N_JOBS.
Deps: numpy, scipy (engine), pandas+tkinter (apatite_gui import).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO = os.environ.get("OKTEDI_REPO") or str(_SCRIPT_DIR.parent)
if not os.path.isdir(REPO):
    sys.exit(f"engine repo not found; set OKTEDI_REPO (tried {REPO})")
sys.path.insert(0, REPO)
import apatite_inversion as inv     # noqa: E402  (thin; safe for child re-import)

# Data + outputs live outside git; point OT_DATA_DIR at the run directory.
# (The observed-apatite CSV and the B2 T-F curve ship with the manuscript
# supplement, not with this repo.)
_data_dir = os.environ.get("OT_DATA_DIR")
if not _data_dir:
    sys.exit("OT_DATA_DIR not set: point it at the run dir holding "
             "test_OT.csv + OT-test-26.6.25_1000-750-B2.csv")
HERE = Path(_data_dir)
TF_PATH = HERE / "OT-test-26.6.25_1000-750-B2.csv"   # byte-identical to 26.7.7's
OBS_PATH = HERE / "test_OT.csv"                      # byte-identical to 26.7.7's

# ==================== OT-APA-TEST-3 manifest state (2026-08-09) ==============
# 9-param hand target (the GUI eye-fit sliders at save time = run_014 params).
TARGET9 = dict(
    H2O_init=4.04678, Cl_init=0.41928, F_init=0.35405,
    D_xl_m_Cl=0.2103, D_xl_m_F=1.45958, D_xl_m_H2O=0.17762,
    D_fl_m_Cl=25.95429, D_fl_m_F=2.99798, H2O_Sat=11.01844,
)
K_SPEC = 0.388                       # manifest k_spec (26.7.7 canonical: 0.463)
# The manifest bounds are a +/-30% box around the hand target (H2O_Sat capped
# at 12). The converged (ftol=1e-10) reference run RAILED on them: accepted-
# family medians pinned at the D_xl_m_Cl and H2O_Sat UBs, and 23% of the
# top-5% members sat within 2% of the D_fl_m_Cl UB (central D_fl_m_Cl 33.09 vs
# cap 33.74). User ruling 2026-08-09 night: WIDEN D_fl_m_Cl UB -> 45 (26.7.7
# precedent used 10-45) and D_xl_m_Cl UB -> 0.35; KEEP the H2O_Sat 12 cap as a
# deliberate physical depth prior (12 wt% already extrapolates to ~400 MPa).
# OT_BOUNDS=manifest reproduces the original box (golden mode); default wide.
MANIFEST_BOUNDS = dict(
    H2O_init=(2.8327, 5.2608), Cl_init=(0.2935, 0.5451), F_init=(0.2478, 0.4603),
    D_xl_m_Cl=(0.1472, 0.2734), D_xl_m_F=(1.0217, 1.8975), D_xl_m_H2O=(0.1243, 0.3),
    D_fl_m_Cl=(18.168, 33.7406), D_fl_m_F=(2.0986, 3.8974), H2O_Sat=(7.7129, 12.0),
)
WIDE_BOUNDS = dict(MANIFEST_BOUNDS,
                   D_xl_m_Cl=(0.1472, 0.35), D_fl_m_Cl=(18.168, 45.0),
                   # rail round 2 (2026-08-10): the first widened converged run
                   # cleared the round-1 rails but piled >30% of the accepted
                   # family - and the convergence attractor itself - onto the
                   # D_fl_m_F UB 3.8974 (a +/-30%-box artefact; the 26.7.7
                   # canonical run's own UB was 5.0). Restored to 5.0 per the
                   # same ruling logic; H2O_Sat stays capped at 12 (physical).
                   D_fl_m_F=(2.0986, 5.0))
BOUNDS_MODE = os.environ.get("OT_BOUNDS", "wide").lower()
BOUNDS = MANIFEST_BOUNDS if BOUNDS_MODE == "manifest" else WIDE_BOUNDS
NUM_START = int(os.environ.get("NUM_START", "2000"))
N_JOBS = int(os.environ.get("N_JOBS", "13"))         # manifest n_jobs
FTOL = float(os.environ.get("OT_FTOL", "0.01"))   # manifest ms_ftol (see caveat);
# OT_FTOL=1e-10 runs the fully-converged REFERENCE variant (band re-ruling aid)
METRIC = "logrmse"                   # manifest ms_metric
SEED = 0                             # inv.SEED — same stream as the GUI
RMSE_MULT = 5.0                      # accepted family (user ruling 2026-08-09)
GOLDEN_BEST = 0.00864008234041       # run_014 best_rmse (full-n runs only)
NUM_FRAC_TARGET = 200                # GUI target grid (self._grid(200))
# =============================================================================

PARAM_ORDER = inv.PARAM_ORDER


def gui_grid(tf, n):
    """EXACT copy of apatite_gui._grid's path-mode branch: re-interpolate the
    NUM_F_LIVE-pt loaded curve to n points (double interpolation — see
    docstring; do NOT 'simplify' to am.build_F_and_T(raw, n))."""
    F, T_K, _ = tf
    if len(F) != n:
        F2 = np.linspace(F.max(), max(0.1, F.min()), n)
        order = np.argsort(F)
        return F2, np.interp(F2, F[order], T_K[order])
    return F, T_K


def main():
    if not TF_PATH.exists():
        sys.exit(f"B2 T-F curve not found: {TF_PATH}\n  set OT_DATA_DIR to the run dir.")
    # heavy import kept out of module level so ProcessPoolExecutor children
    # (which re-import __main__ on Windows spawn) stay cheap
    import apatite_gui as ag

    # ---------- Step 1: target curve, exactly as the GUI forwards it ----------
    tf = ag.load_tf_curve(str(TF_PATH))              # 120-pt regrid of the B2 csv
    F_t, T_t = gui_grid(tf, NUM_FRAC_TARGET)
    st = ag.compute_state(TARGET9, K_SPEC, F_t, T_t)
    if not st["conv"]:
        sys.exit("[target] fixed-point forward did NOT converge at the manifest params")
    F_fit, T_fit = gui_grid(tf, ag.NUM_F_FIT)
    inv.write_target(dict(Frac_Melt=st["F"], XCl_div_XOH=st["cloh"],
                          XF_div_XOH=st["foh"], XF_div_XCl=st["fcl"]),
                     str(HERE / "OT_inv_target_curve.csv"))
    print("========== Step 1: OT-APA-TEST-3 target curve (GUI-equivalent) ==========")
    print(f"  grids: tf {len(tf[0])} pts -> target {len(F_t)} / fit {len(F_fit)}"
          f"  T {T_t[0]-273.15:.0f}->{T_t[-1]-273.15:.0f} C  F {F_t[0]:.3f}->{F_t[-1]:.3f}")
    print(f"  k_spec {K_SPEC}  target Cl/OH {st['cloh'].min():.3g}..{st['cloh'].max():.3g}"
          f"  F/OH {st['foh'].min():.3g}..{st['foh'].max():.3g}")

    # engine self-check on THIS grid (fixed-point vs least_squares agreement)
    inv.TARGET_PARAMS = ag.full_params(TARGET9)
    inv.BOUNDS = BOUNDS
    inv.K_SPECIATION, inv.NUM_START, inv.SEED = K_SPEC, NUM_START, SEED
    inv._self_check(F_fit, T_fit)

    # ---------- Step 2: multistart, exactly as gui._ms_worker runs it ----------
    lb = np.array([BOUNDS[k][0] for k in PARAM_ORDER])
    ub = np.array([BOUNDS[k][1] for k in PARAM_ORDER])
    bounds_list = list(zip(lb, ub))
    rng = np.random.default_rng(SEED)
    starts = [lb + rng.random(len(lb)) * (ub - lb) for _ in range(NUM_START)]
    lsopts = dict(inv._LSOPTS, ftol=FTOL)
    init_args = (F_fit, T_fit, st["F"], st["cloh"], st["foh"], bounds_list,
                 K_SPEC, 0.0, lsopts, METRIC)
    nj = max(1, min(N_JOBS, NUM_START))
    print(f"\n========== Step 2: multistart ({NUM_START} starts, {nj} procs, "
          f"metric={METRIC}, ftol={FTOL:g}) ==========")
    t0 = time.time()
    results = []
    if nj == 1:
        inv._worker_init(*init_args)
        for i, x0 in enumerate(starts):
            results.append(inv._fit_one(x0))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{NUM_START}  [{time.time()-t0:.0f}s]", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=nj, initializer=inv._worker_init,
                                 initargs=init_args) as ex:
            futs = [ex.submit(inv._fit_one, x0) for x0 in starts]
            for i, fut in enumerate(as_completed(futs)):
                results.append(fut.result())
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{NUM_START}  [{time.time()-t0:.0f}s]", flush=True)
    results.sort(key=lambda t: t[0])
    print(f"  done in {time.time()-t0:.1f}s")
    inv.write_multioutput(results, str(HERE / "OT_inv_multistart.csv"))

    rmses = np.array([r[0] for r in results])
    X = np.array([r[1] for r in results])
    best = rmses[0]

    # ---------- GOLDEN: reproduce the GUI's own run_014 ----------
    # only comparable under the EXACT run_014 config: manifest bounds (starts
    # are drawn from the bounds box, so widening changes every start), n=2000,
    # ftol=0.01
    if NUM_START == 2000 and FTOL == 0.01 and BOUNDS == MANIFEST_BOUNDS:
        rel = abs(best - GOLDEN_BEST) / GOLDEN_BEST
        tag = "PASS (exact)" if rel < 1e-12 else (
            f"PASS (rel {rel:.2e} — scipy/BLAS provenance)" if rel < 1e-6
            else f"FAIL (rel {rel:.2e})")
        print(f"  [golden vs run_014] best {best:.12g} vs {GOLDEN_BEST:.12g} -> {tag}")

    cutoff = best * RMSE_MULT
    acc = rmses <= cutoff
    print(f"  RMSE: best={best:.4g}  median={np.median(rmses):.4g}  max={rmses.max():.4g}")
    print(f"  accepted family: RMSE <= {cutoff:.4g} (5x best)  ->  {int(acc.sum())}/{len(rmses)} fits")
    print(f"\n  {'param':11s}{'lb':>8s}{'ub':>8s}{'min':>9s}{'med':>9s}{'max':>9s}{'rail':>6s}")
    print("  " + "-" * 58)
    for i, k in enumerate(PARAM_ORDER):
        blo, bhi = BOUNDS[k]
        col = X[acc, i]
        span = bhi - blo
        rail = ("LB" if np.mean(col <= blo + 0.02 * span) > 0.3 else
                ("UB" if np.mean(col >= bhi - 0.02 * span) > 0.3 else "-"))
        print(f"  {k:11s}{blo:8.3g}{bhi:8.3g}{col.min():9.4g}{np.median(col):9.4g}"
              f"{col.max():9.4g}{rail:>6s}")
    print("  (rail = >30% of accepted fits pinned within 2% of a bound -> widen it)")

    h2o, cl, f = X[acc, 0], X[acc, 1], X[acc, 2]
    print(f"\n========== melt initial volatiles — accepted family (n={int(acc.sum())}) ==========")
    print(f"  H2O wt%: {h2o.min():.2f} .. {h2o.max():.2f}   (median {np.median(h2o):.2f})")
    print(f"  Cl  wt%: {cl.min():.3f} .. {cl.max():.3f}   (median {np.median(cl):.3f})")
    print(f"  F   wt%: {f.min():.3f} .. {f.max():.3f}   (median {np.median(f):.3f})")
    print(f"  H2O_Sat: {X[acc, 8].min():.2f} .. {X[acc, 8].max():.2f}   "
          f"(median {np.median(X[acc, 8]):.2f})  <- would-be declaration")

    # ---------- top-5% by logRMSE (user ruling 2026-08-09 night: the manuscript
    # CENTRAL = median of the best 5% of starts, replacing the accepted-family
    # componentwise median; results are already sorted ascending) ----------
    n_top5 = max(2, int(round(0.05 * len(results))))
    Xt5 = X[:n_top5]
    t5_med = np.median(Xt5, axis=0)
    print(f"\n========== top-5% by {METRIC} (n={n_top5}, misfit 1.000-"
          f"{rmses[n_top5 - 1] / best:.4f}x best) ==========")
    print(f"  {'param':11s}{'min':>9s}{'MEDIAN':>9s}{'max':>9s}")
    for i, k in enumerate(PARAM_ORDER):
        print(f"  {k:11s}{Xt5[:, i].min():9.4g}{t5_med[i]:9.4g}{Xt5[:, i].max():9.4g}")

    # ---------- bestfit + family-median param files (26.7.7 format) ----------
    import csv
    best_params = inv.coeff_to_params(X[0], prim_boil=0.0)
    bf = HERE / "OT_inv_bestfit_params.csv"
    with open(bf, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(best_params.keys()) + ["K_spec", "RMSE"])
        w.writerow([f"{best_params[k]:.6g}" for k in best_params]
                   + [f"{K_SPEC:g}", f"{best:.6g}"])
    med_params = inv.coeff_to_params(np.median(X[acc], axis=0), prim_boil=0.0)
    # RMSE_central = the median-params curve's OWN misfit, evaluated with the
    # SAME grids/metric the multistart used. The overlay's panels band anchors
    # its |RMSE - RMSE_central| window on this value when the componentwise
    # median is NOT itself a family member (26.7.7's 1-D ridge made it one to
    # rel 1e-3; this run's loose-ftol fan does not — closest rel.dev ~0.29).
    med_x = np.array([med_params[k] for k in PARAM_ORDER])
    rmse_med = float(inv.rmse_to_target(med_x, F_fit, T_fit, st["F"], st["cloh"],
                                        st["foh"], K_SPEC, 0.0, metric=METRIC))
    print(f"  central (median-params) own misfit: {rmse_med:.6g} "
          f"= {rmse_med / best:.3f}x best  (rank-equivalent "
          f"{int((rmses < rmse_med).sum()) + 1}/{len(rmses)})")
    mf = HERE / "OT_inv_median_params.csv"
    with open(mf, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(med_params.keys())
                   + ["K_spec", "RMSE_best", "n_family", "RMSE_central"])
        w.writerow([f"{med_params[k]:.6g}" for k in med_params]
                   + [f"{K_SPEC:g}", f"{best:.6g}", str(int(acc.sum())),
                      f"{rmse_med:.12g}"])

    # top-5% median params file — the manuscript-central artifact per the
    # 2026-08-09-night ruling (same schema as the family-median file, so any
    # downstream consumer can be pointed at either)
    t5_params = inv.coeff_to_params(t5_med, prim_boil=0.0)
    rmse_t5 = float(inv.rmse_to_target(t5_med, F_fit, T_fit, st["F"], st["cloh"],
                                       st["foh"], K_SPEC, 0.0, metric=METRIC))
    print(f"  top-5% median own misfit: {rmse_t5:.6g} = {rmse_t5 / best:.3f}x best")
    tf5 = HERE / "OT_inv_top5_median_params.csv"
    with open(tf5, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(t5_params.keys())
                   + ["K_spec", "RMSE_best", "n_family", "RMSE_central"])
        w.writerow([f"{t5_params[k]:.6g}" for k in t5_params]
                   + [f"{K_SPEC:g}", f"{best:.6g}", str(n_top5),
                      f"{rmse_t5:.12g}"])
    print(f"\n[written] OT_inv_target_curve.csv, OT_inv_multistart.csv, {bf.name}, "
          f"{mf.name} (family median), {tf5.name} (top-5% median — manuscript "
          f"central per ruling 2026-08-09; bounds mode: {BOUNDS_MODE})")


if __name__ == "__main__":
    main()
