#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_melts_lld.py
==================
Drive rhyolite-MELTS (alphaMELTS for Python) to produce a fractional-crystallisation
LIQUID LINE OF DESCENT for the Ok Tedi melt proxy, and write the CSV consumed by
magma_viscosity_evolution.py.

Agreed defaults (Ok Tedi P1):
  model      : rhyolite-MELTS 1.0.2
  bulk       : whole-rock majors read from an external compositions CSV
               (OT_BULK env var, else `compositions.csv` next to this script;
               the Ok Tedi compositions ship with the manuscript supplement),
               LOI-free renormalised to 100 wt% + initial H2O (optional H2O
               column, default 4.0 wt%); total Fe as Fe2O3, ferric/ferrous
               set by the fO2 buffer (Kress & Carmichael 1991).
  fO2        : FMQ + 0.3   (Log fO2 Path: FMQ ; Log fO2 Offset: 0.3 ; cf. zircon dFMQ +0.26)
  pressure   : 2000 bar (200 MPa), isobaric
  mode       : Fractionate Solids   (matches Lormand's Rayleigh assumption)
  T path     : liquidus -> down in 5 C steps; stop at F < 0.10 or T < 700 C

Usage:
  python build_melts_lld.py                # full run (first CSV row), writes
                                           #   oktedi_melts_lld_<sample>_<date>.csv
  python build_melts_lld.py --sample <id>  # pick another row of the CSV
  python build_melts_lld.py --trial        # 3-step verbose smoke test (no output suppression)

CAVEATS (see README / plan):
  * rhyolite-MELTS has NO amphibole/biotite solution model -> LLD biased for this
    hydrous, oxidised intermediate magma. Cross-check with Petrolog3 before publishing.
  * No apatite phase -> P2O5 accumulates unrealistically (minor for viscosity).
  * P, fO2, initial H2O are chosen defaults -> run a sensitivity grid later.
"""

import csv
import math
import os
import sys
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
# alphaMELTS-for-Python is not bundled: point at it via the ALPHAMELTS_PY env
# var (default: an `alphamelts-py` checkout next to this repo).
sys.path.insert(0, os.environ.get("ALPHAMELTS_PY")
                or os.path.join(HERE, "..", "alphamelts-py"))
from alphamelts_loader import MELTSdynamic  # noqa: E402

# --- inputs -----------------------------------------------------------------
# Whole-rock XRF majors are NOT bundled with this repo (they ship with the
# manuscript supplement). Provide them as a CSV — path from the OT_BULK env
# var, else `compositions.csv` next to this script — with a header row
#   sample,SiO2,TiO2,Al2O3,Fe2O3,MnO,MgO,CaO,Na2O,K2O,P2O5[,LOI][,H2O]
# ('#' comment lines allowed). RAW wt%; total Fe as Fe2O3 (TFe2O3) — MELTS
# takes the total and the fO2 buffer sets the ferric/ferrous split (Kress &
# Carmichael 1991), so no manual Fe partitioning here. LOI (if present) is
# excluded and the majors are renormalised anhydrous to 100 wt% by
# make_bulk(); initial melt H2O comes from the optional H2O column (else
# H2O_INIT). The pre/post sums are printed as a self-check at run time.
BULK_CSV = os.environ.get("OT_BULK") or os.path.join(HERE, "compositions.csv")
H2O_INIT = 4.0       # wt% initial melt H2O default (protocol init; NOT the whole-rock LOI)


def load_samples(path=None):
    """Read the compositions CSV -> {sample_id: {column: wt%, ...}} in file order."""
    path = path or BULK_CSV
    if not os.path.exists(path):
        sys.exit(f"bulk-composition CSV not found: {path}\n"
                 f"  Set OT_BULK to a whole-rock majors CSV (header: sample,"
                 f"SiO2,...,P2O5[,LOI][,H2O]; RAW wt%). The Ok Tedi "
                 f"compositions ship with the manuscript supplement.")
    samples = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(line for line in fh if not line.lstrip().startswith("#"))
        for row in rdr:
            sid = (row.pop("sample", None) or "").strip()
            if not sid:
                continue
            samples[sid] = {k.strip(): float(v) for k, v in row.items()
                            if k and v not in (None, "")}
    if not samples:
        sys.exit(f"no sample rows found in {path}")
    return samples


def make_bulk(sample_id, samples):
    """LOI-free renormalisation to 100 wt% + initial H2O. Returns (bulk,
    anhydrous raw sum, LOI) so the caller can print the normalisation
    self-check."""
    raw = samples[sample_id]
    majors = {k: v for k, v in raw.items() if k not in ("LOI", "H2O")}
    s = sum(majors.values())
    bulk = {k: 100.0 * v / s for k, v in majors.items()}
    bulk["H2O"] = raw.get("H2O", H2O_INIT)
    return bulk, s, raw.get("LOI", 0.0)


# Loaded in __main__ (module globals; run() reads BULK):
SAMPLES = SAMPLE = BULK = _RAW_SUM = _LOI = None
PRESSURE_BAR = 2000.0
FO2_PATH = "FMQ"
FO2_OFFSET = "0.3"
DT = 5.0
T_START_GUESS = 1300.0
T_FLOOR = 700.0
F_STOP = 0.10
OUT_OXIDES = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO",
              "Na2O", "K2O", "P2O5"]
# Lormand H2O(F) for the consistency diagnostic (prim. boiling off):
LORMAND_H2O_INIT, LORMAND_D_H2O, LORMAND_H2O_SAT = 4.0, 0.12, 8.5


@contextmanager
def _noop():
    yield


@contextmanager
def suppress_c_output():
    """Silence the MELTS C library's verbose stdout/stderr at the fd level."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(devnull); os.close(saved[0]); os.close(saved[1])


def lormand_h2o(F):
    return min(LORMAND_H2O_INIT * F ** (LORMAND_D_H2O - 1.0), LORMAND_H2O_SAT)


def _liq_mass(e):
    m = e.getProperty("mass", "liquid1")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return None
    return float(m)


def solids_present(e):
    """True if any solid phase (not liquid/fluid/water/bulk/oxygen) has mass > 0."""
    if not isinstance(e.mass, dict):
        return False
    for k, v in e.mass.items():
        kl = k.lower()
        if kl in ("bulk", "oxygen"):
            continue
        if kl.startswith(("liquid", "fluid", "water")):
            continue
        if v and not math.isnan(v) and v > 1e-6:
            return True
    return False


def liquid_oxides(e):
    """Return (anhydrous-majors dict normalised to 100, melt H2O wt%) for liquid1,
    with FeOt = FeO + 0.8998*Fe2O3. Robust to disp units (renormalised)."""
    names = e.status.endmembers["bulk"]                 # lowercased oxide names
    vals = e.getProperty("dispComposition", "liquid1")
    raw = {n: float(v) for n, v in zip(names, vals)}
    tot_all = sum(v for v in raw.values() if not math.isnan(v))
    h2o_wt = 100.0 * raw.get("h2o", 0.0) / tot_all if tot_all else float("nan")
    feot = raw.get("feo", 0.0) + 0.8998 * raw.get("fe2o3", 0.0)
    majors = {"SiO2": raw.get("sio2", 0.0), "TiO2": raw.get("tio2", 0.0),
              "Al2O3": raw.get("al2o3", 0.0), "FeOt": feot, "MnO": raw.get("mno", 0.0),
              "MgO": raw.get("mgo", 0.0), "CaO": raw.get("cao", 0.0),
              "Na2O": raw.get("na2o", 0.0), "K2O": raw.get("k2o", 0.0),
              "P2O5": raw.get("p2o5", 0.0)}
    s = sum(majors.values())
    majors = {k: 100.0 * v / s for k, v in majors.items()}   # anhydrous -> 100
    return majors, h2o_wt


def run(trial=False, stream_path=None):
    # stream_path: append every accepted node to a .part CSV (flushed) so a hard
    # MELTS C-library crash mid-path doesn't lose the nodes already computed
    # (observed at 100 MPa: process dies with no traceback deep in the FC loop).
    stream_fh = None
    if stream_path:
        stream_fh = open(stream_path, "w", newline="", encoding="utf-8")
        stream_fh.write(",".join(["F", "T_C"] + OUT_OXIDES + ["H2O_melt_MELTS"]) + "\n")
        stream_fh.flush()
    cm = _noop if trial else suppress_c_output
    m = MELTSdynamic(1)
    e = m.engine
    for ox, g in BULK.items():
        e.setBulkComposition(ox, g)
    # impose fO2 buffer (verified record names) + pressure
    e.setSystemProperties("Log fO2 Path", FO2_PATH, "Log fO2 Offset", FO2_OFFSET)
    e.pressure = PRESSURE_BAR

    # --- coarse liquidus search (no fractionation): step down until first solids ---
    # M0 = all-liquid system mass (the buffer exchanges O2, so this != sum of inputs).
    t_liq, M0 = None, None
    Tc = T_START_GUESS
    while Tc > T_FLOOR:
        e.temperature = float(Tc)
        with cm():
            e.calcEquilibriumState(1, 1)
        if M0 is None and not solids_present(e):
            M0 = _liq_mass(e)                  # all-liquid system mass (denominator)
        if solids_present(e):
            t_liq = Tc
            break
        Tc -= 10
    if t_liq is None:
        t_liq = math.floor(T_START_GUESS)
    if M0 is None:
        M0 = _liq_mass(e) or sum(BULK.values())
    print(f"[liquidus~] first solids by T = {t_liq:.0f} C "
          f"(coarse 10 C search; M0={M0:.2f} g; P={PRESSURE_BAR:.0f} bar, {FO2_PATH}+{FO2_OFFSET})")

    # --- fractional crystallisation from the liquidus downward ---
    e.setSystemProperties("Mode", "Fractionate Solids")
    rows = []
    T = float(t_liq)
    nmax = 3 if trial else 100000
    n = 0
    while T >= T_FLOOR and n < nmax:
        e.temperature = T
        try:
            with cm():
                e.calcEquilibriumState(1, 1)   # Mode=Fractionate updates bulk to residual
        except Exception as ex:                # noqa: BLE001
            print(f"  [stop] exception at T={T:.0f}: {ex}")
            break
        if e.status.failed:
            print(f"  [stop] MELTS failed at T={T:.0f}")
            break
        m_liq = _liq_mass(e)
        if m_liq is None:
            print(f"  [stop] no liquid at T={T:.0f}")
            break
        F = m_liq / M0
        majors, h2o_melt = liquid_oxides(e)
        rows.append(dict(F=F, T_C=T, h2o_melt=h2o_melt, **majors))
        if stream_fh:
            stream_fh.write(",".join([f"{F:.4f}", f"{T:.1f}"]
                                     + [f"{majors[k]:.3f}" for k in OUT_OXIDES]
                                     + [f"{h2o_melt:.3f}"]) + "\n")
            stream_fh.flush()
        if trial:
            print(f"  T={T:.0f} F={F:.3f} SiO2={majors['SiO2']:.1f} "
                  f"H2O_melt(MELTS)={h2o_melt:.2f} residual_bulk={sum(e.bulkComposition):.2f}g "
                  f"H2O_Lormand={lormand_h2o(F):.2f}")
        if F < F_STOP:
            break
        T -= DT
        n += 1
    if stream_fh:
        stream_fh.close()
    return rows


def write_csv(rows, path):
    # keep strictly decreasing F (drop duplicate/non-monotonic nodes for interpolation)
    clean, last = [], None
    for r in sorted(rows, key=lambda d: d["F"], reverse=True):
        if last is None or r["F"] < last - 1e-6:
            clean.append(r); last = r["F"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        # raw comment line (not via csv.writer, which would quote the commas)
        fh.write(f"# Ok Tedi {SAMPLE} rhyolite-MELTS 1.0.2 fractional LLD "
                 f"(LOI-free renormalised bulk); "
                 f"FMQ+0.3, {PRESSURE_BAR/10:.0f} MPa, init H2O {BULK['H2O']:g} wt%. "
                 f"H2O_melt_MELTS = MELTS melt H2O "
                 f"(diagnostic only; viscosity tool uses Lormand H2O).\n")
        w = csv.writer(fh)
        w.writerow(["F", "T_C"] + OUT_OXIDES + ["H2O_melt_MELTS"])
        for r in clean:
            w.writerow([f"{r['F']:.4f}", f"{r['T_C']:.1f}"]
                       + [f"{r[k]:.3f}" for k in OUT_OXIDES]
                       + [f"{r['h2o_melt']:.3f}"])
    print(f"[written] {path}  ({len(clean)} nodes, "
          f"F {clean[0]['F']:.3f} -> {clean[-1]['F']:.3f})")
    return clean


if __name__ == "__main__":
    import time
    trial = "--trial" in sys.argv
    out_path = None
    SAMPLES = load_samples()
    SAMPLE = next(iter(SAMPLES))       # default = first row of the CSV
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--pbar":              # pressure override (bar) for the P sensitivity grid
            PRESSURE_BAR = float(argv[i + 1])
        elif a == "--out":
            out_path = argv[i + 1]
        elif a == "--sample":          # pick a sample row of the compositions CSV
            SAMPLE = argv[i + 1]
            if SAMPLE not in SAMPLES:
                sys.exit(f"sample {SAMPLE!r} not in {BULK_CSV} "
                         f"(rows: {', '.join(SAMPLES)})")
    BULK, _RAW_SUM, _LOI = make_bulk(SAMPLE, SAMPLES)
    if out_path is None:
        out_path = os.path.join(
            HERE, f"oktedi_melts_lld_{SAMPLE}_{time.strftime('%y%m%d')}.csv")
    # normalisation self-check (acceptance item): raw anhydrous sum -> 100
    print(f"[bulk] {SAMPLE}: raw majors sum {_RAW_SUM:.2f} + LOI {_LOI:.2f} "
          f"= {_RAW_SUM + _LOI:.2f} wt%; LOI-free renormalised sum = "
          f"{sum(v for k, v in BULK.items() if k != 'H2O'):.4f} wt% (+ H2O {BULK['H2O']:g})")
    for ox in ("SiO2", "Al2O3", "Fe2O3", "CaO", "Na2O", "K2O"):
        if ox in BULK:
            print(f"       {ox:6s} {SAMPLES[SAMPLE][ox]:6.2f} -> {BULK[ox]:7.4f}")
    rows = run(trial=trial, stream_path=None if trial else out_path + ".part")
    if not rows:
        print("no rows produced"); sys.exit(1)
    if trial:
        print(f"[trial OK] {len(rows)} steps recorded")
    else:
        clean = write_csv(rows, out_path)
        try:
            os.remove(out_path + ".part")
        except OSError:
            pass
        print("\n[H2O consistency]  F :  MELTS_melt_H2O  vs  Lormand_H2O  (wt%)")
        for r in clean[::max(1, len(clean) // 8)]:
            print(f"   F={r['F']:.2f}   MELTS={r['h2o_melt']:5.2f}   "
                  f"Lormand={lormand_h2o(r['F']):5.2f}")
