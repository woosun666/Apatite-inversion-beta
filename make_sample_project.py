#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sample_project.py — generate SYNTHETIC demo/regression data for the GUI
(P1 spec deliverable 5: sample project with synthetic apatite data).

Writes into ./sample_data/ (gitignored — the GENERATOR is the versioned
artifact, the repo stays code-only):
  sample_obs.csv    : 60 synthetic apatite X-site ratios (test_OT.csv layout,
                      forward model + lognormal noise; 12 rows marked ZIR)
  sample_tf.csv     : Huber b=2 cooling curve 1000->750 C (Var1/Var2 layout)
  sample_epma.csv   : 12 synthetic EPMA wt% rows (generic apatite base
                      analyses, jittered) for the M1 import workflow

Demo: python apatite_gui.py -> New blank session -> load these three files
(obs + T-F on the left panel; EPMA via the M1 tab) -> save as sample.otproj.
Regression: the printed forward/multistart numbers are deterministic (SEED).
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apatite_model as am              # noqa: E402

OUT = os.path.join(HERE, "sample_data")
SEED = 7
TARGET = dict(H2O_init=4.0, Cl_init=0.51, F_init=0.46, D_xl_m_Cl=0.05,
              D_xl_m_F=1.5, D_xl_m_H2O=0.12, D_fl_m_Cl=22.0, D_fl_m_F=2.5,
              H2O_Sat=10.0, D_fl_m_PrimBoilH2O=0.0)
K_SPEC = 0.463
T_INITIAL, T_FINAL, B = 1000.0, 750.0, 2.0
NOISE = 0.10                            # lognormal sigma on the ratios


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- T-F curve (Huber: T(F) = Tf + (Ti-Tf) * F^(1/b)) ----
    F = np.linspace(1.0, 0.1, 200)
    T = T_FINAL + (T_INITIAL - T_FINAL) * F ** (1.0 / B)
    with open(os.path.join(OUT, "sample_tf.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Var1", "Var2"])
        for t, f in zip(T, F):
            w.writerow([f"{t:.2f}", f"{f:.4f}"])

    # ---- synthetic observed ratios: forward curve + lognormal noise ----
    td, _ = am.forward_model(F, T + 273.15, TARGET, K_SPEC)
    idx = rng.choice(len(F), size=60, replace=False)
    with open(os.path.join(OUT, "sample_obs.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Name", "Rock type", "F/OH", "Cl/OH", "F/Cl", "name"])
        for n, i in enumerate(sorted(idx)):
            foh = td["XF_div_XOH"][i] * rng.lognormal(0, NOISE)
            cloh = td["XCl_div_XOH"][i] * rng.lognormal(0, NOISE)
            rock = "SYN-ZIR" if n % 5 == 0 else "SYN-matrix"   # 12 ZIR rows
            nm = f"SYN-{n+1:02d}"
            w.writerow([nm, rock, f"{foh:.4f}", f"{cloh:.5f}",
                        f"{foh/cloh:.3f}", nm])

    # ---- synthetic EPMA rows: generic apatite base analyses (SYNTHETIC,
    # typical igneous apatite ranges — not measured data), jittered ----
    golden = [
        dict(F=1.50, Cl=1.10, CaO=54.5, Na2O=0.10, MgO=0.12, MnO=0.15,
             FeO=0.30, SrO=0.06, P2O5=41.5, SiO2=0.15, SO3=0.12),
        dict(F=2.60, Cl=0.85, CaO=54.0, Na2O=0.05, MgO=0.30, MnO=0.20,
             FeO=0.35, SrO=0.05, P2O5=41.2, SiO2=0.25, SO3=0.08),
        dict(F=2.20, Cl=0.55, CaO=54.8, Na2O=0.08, MgO=0.20, MnO=0.10,
             FeO=0.25, SrO=0.07, P2O5=41.8, SiO2=0.10, SO3=0.15),
        dict(F=1.90, Cl=0.95, CaO=54.2, Na2O=0.07, MgO=0.25, MnO=0.18,
             FeO=0.32, SrO=0.06, P2O5=41.3, SiO2=0.20, SO3=0.10),
    ]
    cols = ["Comment", "F", "Cl", "CaO", "Na2O", "MgO", "MnO", "FeO", "SrO",
            "P2O5", "SiO2", "SO3", "Total"]
    with open(os.path.join(OUT, "sample_epma.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for n in range(12):
            g = golden[n % len(golden)]
            j = lambda v, s=0.03: v * float(rng.lognormal(0, s))  # noqa: E731
            row = dict(Comment=f"SYNAP-{n+1:02d}", F=j(g["F"]), Cl=j(g["Cl"]),
                       CaO=j(g["CaO"], 0.005), Na2O=j(g["Na2O"]),
                       MgO=j(g["MgO"]), MnO=j(g["MnO"]), FeO=j(g["FeO"]),
                       SrO=j(g["SrO"]), P2O5=j(g["P2O5"], 0.005),
                       SiO2=j(g["SiO2"]), SO3=j(g["SO3"]))
            row["Total"] = sum(v for k, v in row.items() if k != "Comment")
            w.writerow([row[c] if c == "Comment" else f"{row[c]:.4f}"
                        for c in cols])

    print(f"[written] {OUT}: sample_obs.csv (60 pts, 12 ZIR), sample_tf.csv "
          f"(b={B:g}), sample_epma.csv (12 rows)")
    print("open the GUI -> New blank session -> load these; see README §GUI.")


if __name__ == "__main__":
    main()
