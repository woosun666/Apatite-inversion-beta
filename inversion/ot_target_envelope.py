#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OT apatite inversion — melt-volatile ENVELOPE for the manuscript canonical
(OT-APA-TEST-3 hand target, k_spec 0.388; bracket method per the
2026-07-05 ruling, adapted from the earlier envelope protocol).

Why: the multistart family around ONE target quantifies parameter trade-off,
NOT data uncertainty (the objective fits the hand-set target curve; the 147
apatite points are not in it). The honest melt-volatile range = the UNION of
the families recovered from low / central / high target curves that BRACKET
the observed cloud. The bracket is EYE-VALIDATED (preview mode).

26.8.10 adaptation decisions (2026-08-10):
  * CENTRAL = the OT-APA-TEST-3.otproj hand target (= the canonical run's
    target; k_spec 0.388).
  * LOW/HIGH factors = the USER'S OWN bracket ratios, recovered from the
    .otproj env_targets block: Cl x0.800 / x1.300, F x0.720 / x1.300.
    (The manifest trio itself is NOT used verbatim: it was set 2026-08-06
    around an older central — H2O 4.397, D_xl_m_Cl 0.0903 — and is
    inconsistent with the canonical target; the user's RATIOS applied to the
    CURRENT central keep their bracket-width judgment. Eye-check the preview
    figure before trusting the full run.)
  * BOUNDS = the wide2 canonical box, further opened on Cl_init / F_init so
    all three targets sit INSIDE (envelope must not be clipped): the HIGH
    target lands exactly on the wide2 Cl/F UBs (0.545 / 0.460), so those open
    to 0.71 / 0.60 (26.7.7-envelope style). Engine invariant kept:
    H2O_init UB (5.2608) < H2O_Sat LB (7.7129).
  * ftol = engine default (1e-10, fully converged) — the envelope keeps its
    own 26.7.7 convention; the two-tolerance protocol governs the CENTRAL and
    the section-5.3 BAND, not this instrument.
  * grids = am.build_F_and_T directly (single interpolation), faithful to the
    26.7.7 envelope script — NOT the GUI double-interp of the canonical
    wrapper (protocol-internal consistency wins here).

MODE (argv[1]): preview (default) -> targets + overplot only; full -> + multistart.
Env: OT_DATA_DIR (data+outputs; needs test_OT.csv + the B2 csv), OKTEDI_REPO,
     NUM_START, OT_METRIC (default logrmse).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
# pandas / matplotlib imported lazily (Windows 'spawn' workers re-import this file).

_SCRIPT_DIR = Path(__file__).resolve().parent
_data_dir = os.environ.get("OT_DATA_DIR")
if not _data_dir:
    sys.exit("OT_DATA_DIR not set: point it at the data+output dir holding "
             "test_OT.csv + OT-test-26.6.25_1000-750-B2.csv")
HERE = Path(_data_dir)
OBS_PATH = HERE / "test_OT.csv"
TF_PATH = HERE / "OT-test-26.6.25_1000-750-B2.csv"
REPO = os.environ.get("OKTEDI_REPO") or str(_SCRIPT_DIR.parent)
if not os.path.isdir(REPO):
    sys.exit(f"engine repo not found; set OKTEDI_REPO (tried {REPO})")
sys.path.insert(0, REPO)
import apatite_model as am          # noqa: E402
import apatite_inversion as inv     # noqa: E402

# ---- central target = the OT-APA-TEST-3 / 26.8.10 canonical hand target ----
CENTRAL = dict(H2O_init=4.04678, Cl_init=0.41928, F_init=0.35405,
               D_xl_m_Cl=0.2103, D_xl_m_F=1.45958, D_xl_m_H2O=0.17762,
               D_fl_m_Cl=25.95429, D_fl_m_F=2.99798, H2O_Sat=11.01844,
               D_fl_m_PrimBoilH2O=0.0)


def _scaled(base, fF, fCl):
    d = dict(base)
    d["F_init"] = base["F_init"] * fF
    d["Cl_init"] = base["Cl_init"] * fCl
    return d


# LOW / HIGH brackets: the user's own env_targets ratios (see docstring).
LOW = _scaled(CENTRAL, 0.72, 0.80)
HIGH = _scaled(CENTRAL, 1.30, 1.30)
TARGETS = {"low": LOW, "central": CENTRAL, "high": HIGH}

K_SPEC = 0.388
NUM_FRAC_TARGET = 200
NUM_FRAC_FIT = 100
# WIDE bounds (envelope must NOT be clipped) = wide2 canonical box, opened on
# Cl_init / F_init to contain all 3 targets (see docstring).
BOUNDS = dict(H2O_init=(2.8327, 5.2608), Cl_init=(0.25, 0.71), F_init=(0.20, 0.60),
              D_xl_m_Cl=(0.1472, 0.35), D_xl_m_F=(1.0217, 1.8975),
              D_xl_m_H2O=(0.1243, 0.3), D_fl_m_Cl=(18.168, 45.0),
              D_fl_m_F=(2.0986, 5.0), H2O_Sat=(7.7129, 12.0))
NUM_START = int(os.environ.get("NUM_START", "120"))
METRIC = os.environ.get("OT_METRIC", "logrmse")
SEED = 0
RMSE_MULT = 5.0
N_JOBS = max(1, round(0.8 * (os.cpu_count() or 1)))
COL = {"low": "#0072B2", "central": "#000000", "high": "#D55E00"}


def build_FT(n):
    # NON-isothermal: T(F) from the B2 cooling curve (Var1=T_C, Var2=F).
    return am.build_F_and_T(str(TF_PATH), None, n)


def load_obs():
    import pandas as pd
    df = pd.read_csv(OBS_PATH)
    for c in ("F/OH", "Cl/OH", "F/Cl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["F/OH", "Cl/OH", "F/Cl"]).copy()
    df = df[df["Rock type"].astype(str).str.strip() != ""].copy()
    df["is_zir"] = df["Rock type"].astype(str).str.contains("ZIR", case=False, na=False)
    return df


def forward_curve(tgt):
    Ft, Tt = build_FT(NUM_FRAC_TARGET)
    td, _ = am.forward_model(Ft, Tt, tgt, K_SPEC)
    return td


def overplot(curves, obs, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mx, zr = obs[~obs["is_zir"]], obs[obs["is_zir"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.7))
    for a, (yc, oc, yl) in zip(ax, [("XF_div_XOH", "F/OH", "XF/XOH"),
                                    ("XF_div_XCl", "F/Cl", "XF/XCl")]):
        for name, td in curves.items():
            a.plot(td["XCl_div_XOH"], td[yc], "-", color=COL[name], lw=1.8,
                   zorder=4, label=f"{name} target")
        a.scatter(mx["Cl/OH"], mx[oc], s=12, marker="o", facecolor="#56B4E9",
                  edgecolor="white", linewidth=0.3, alpha=0.7, zorder=3,
                  label=f"matrix (n={len(mx)})")
        a.scatter(zr["Cl/OH"], zr[oc], s=26, marker="^", facecolor="#009E73",
                  edgecolor="black", linewidth=0.4, alpha=0.8, zorder=3,
                  label=f"ZIR (n={len(zr)})")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set(xlabel="XCl/XOH (apatite Cl/OH)", ylabel=f"{yl} (apatite {oc})",
              title=f"{yl} vs XCl/XOH")
        a.grid(True, which="both", alpha=0.25, lw=0.5)
    ax[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(Path(fname).with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    obs = load_obs()
    curves = {k: forward_curve(t) for k, t in TARGETS.items()}
    mx = obs[~obs["is_zir"]]
    Ft, Tt = build_FT(NUM_FRAC_TARGET)
    print(f"=== 26.8.10 envelope (non-isothermal B2 {Tt[0]-273.15:.0f}->{Tt[-1]-273.15:.0f} C, "
          f"metric={METRIC}, k_spec={K_SPEC}) ===")
    print("=== target volatiles (input) + curve span ===")
    for k, t in TARGETS.items():
        print(f"  {k:8s} H2O {t['H2O_init']:.2f}  Cl {t['Cl_init']:.3f}  F {t['F_init']:.3f}"
              f"   |  curve Cl/OH {curves[k]['XCl_div_XOH'].min():.3g}..{curves[k]['XCl_div_XOH'].max():.3g}"
              f"  F/OH {curves[k]['XF_div_XOH'].min():.3g}..{curves[k]['XF_div_XOH'].max():.3g}")
    print(f"  obs(all) Cl/OH {obs['Cl/OH'].min():.3g}..{obs['Cl/OH'].max():.3g}"
          f"  F/OH {obs['F/OH'].min():.3g}..{obs['F/OH'].max():.3g}"
          f"  (n={len(obs)}; matrix {len(mx)} + ZIR {len(obs) - len(mx)})")
    overplot(curves, obs, str(HERE / "OT_envelope_targets"),
             "OT 26.8.10 envelope — low/central/high bracketing curves vs apatite data "
             f"(non-isothermal B2, {METRIC}, user bracket ratios on the OT-APA-TEST-3 target)")
    print("[written] OT_envelope_targets.png/.pdf")
    if mode != "full":
        print("[preview] inspect bracketing; tune LOW/HIGH factors, then: python ot_target_envelope.py full")
        return

    # ---------- multistart each target, union ----------
    F_fit, T_fit = build_FT(NUM_FRAC_FIT)
    inv.BOUNDS, inv.K_SPECIATION, inv.NUM_START, inv.SEED = BOUNDS, K_SPEC, NUM_START, SEED
    allX = {}
    for k, t in TARGETS.items():
        inv.TARGET_PARAMS = t
        td = curves[k]
        print(f"\n=== multistart [{k}] ({NUM_START} starts, {N_JOBS} procs, metric={METRIC}) ===")
        t0 = time.time()
        res = inv.run_multistart(td["Frac_Melt"], td["XCl_div_XOH"], td["XF_div_XOH"],
                                 F_fit, T_fit, NUM_START, n_jobs=N_JOBS, metric=METRIC)
        inv.write_multioutput(res, str(HERE / f"OT_env_multistart_{k}.csv"))
        rm = np.array([r[0] for r in res])
        X = np.array([r[1] for r in res])
        acc = rm <= RMSE_MULT * rm[0]
        allX[k] = X[acc]
        h, c, f = X[acc, 0], X[acc, 1], X[acc, 2]
        print(f"  [{time.time() - t0:.1f}s] best {METRIC} {rm[0]:.4g}  accepted {int(acc.sum())}/{len(rm)}")
        print(f"  H2O {h.min():.2f}-{h.max():.2f}  Cl {c.min():.3f}-{c.max():.3f}  F {f.min():.3f}-{f.max():.3f}")

    U = np.vstack(list(allX.values()))
    print("\n========== UNION — honest melt-volatile range across the 3 bracketing targets ==========")
    for i, nm, fmt in ((0, "H2O", "%.2f"), (1, "Cl", "%.3f"), (2, "F", "%.3f")):
        col = U[:, i]
        print(f"  {nm:3s} wt%: {fmt % col.min()} .. {fmt % col.max()}   (median {fmt % np.median(col)})")
    np.save(str(HERE / "OT_env_union.npy"), U)
    print("\n[written] OT_env_multistart_{low,central,high}.csv, OT_env_union.npy")


if __name__ == "__main__":
    main()
