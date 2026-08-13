"""
Giordano, Russell & Dingwell (2008) EPSL 271, 123-134
"Viscosity of magmatic liquids: A model"

VFT:  log10(eta[Pa s]) = A + B / (T[K] - C),  A = -4.55 (constant)
B, C computed from melt composition in MOL% (oxides + H2O + F2O_-1).
Model is at 1 atm; pressure-INDEPENDENT.

Use for P1 (Ok Tedi): EARLY-MELT minimum viscosity (pre-degassing),
consistent with the ZHAI window. Inputs all already constrained in P1:
melt majors (whole-rock proxy or de-spiked ZHMI), H2O (~4 wt%, model input),
F (~0.45 wt%, ZHAI forward model), T (Ti-in-zircon).

NB: this is MELT viscosity, NOT magma (suspension) viscosity. Crystal/bubble
loading (Costa 2005 / Mader 2013) -> bulk magma viscosity -> P3.
"""
import math
import numpy as np

# molar masses (g/mol)
MM = dict(SiO2=60.084, TiO2=79.866, Al2O3=101.961, FeO=71.844, MnO=70.937,
          MgO=40.304, CaO=56.077, Na2O=61.979, K2O=94.196, P2O5=141.945,
          H2O=18.015, F=18.998)

# Table 1 coefficients
b1, b2, b3, b4, b5, b6, b7 = 159.6, -173.3, 72.1, 75.7, -39.0, -84.1, 141.5
b11, b12, b13 = -2.43, -0.91, 17.6
c1, c2, c3, c4, c5, c6, c11 = 2.75, 15.7, 8.3, 10.2, -12.3, -99.5, 0.30
A = -4.55

OXIDES = ["SiO2", "TiO2", "Al2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5"]


def to_molpct(wt):
    """wt: dict of wt% for oxides + 'H2O' + 'F' (F as element wt%).
    F enters as the F2O_-1 component: n(F2O_-1) = n(F)/2.
    Returns mol% dict including H2O and F2O_1, normalized to 100."""
    n = {}
    for ox in OXIDES:
        n[ox] = wt.get(ox, 0.0) / MM[ox]
    n["H2O"] = wt.get("H2O", 0.0) / MM["H2O"]
    n["F2O_1"] = (wt.get("F", 0.0) / MM["F"]) / 2.0
    tot = sum(n.values())
    return {k: 100.0 * v / tot for k, v in n.items()}


def BC(m):
    """B, C from mol% dict m."""
    V = m["H2O"] + m["F2O_1"]
    TA = m["TiO2"] + m["Al2O3"]
    FM = m["FeO"] + m["MnO"] + m["MgO"]
    NK = m["Na2O"] + m["K2O"]
    B = (b1 * (m["SiO2"] + m["TiO2"])
         + b2 * (m["Al2O3"])
         + b3 * (m["FeO"] + m["MnO"] + m["P2O5"])
         + b4 * (m["MgO"])
         + b5 * (m["CaO"])
         + b6 * (m["Na2O"] + V)
         + b7 * (V + math.log(1.0 + m["H2O"]))
         + b11 * ((m["SiO2"] + m["TiO2"]) * FM)
         + b12 * ((m["SiO2"] + TA + m["P2O5"]) * (NK + m["H2O"]))
         + b13 * ((m["Al2O3"]) * NK))
    C = (c1 * (m["SiO2"])
         + c2 * TA
         + c3 * FM
         + c4 * (m["CaO"])
         + c5 * NK
         + c6 * (math.log(1.0 + V))
         + c11 * ((m["Al2O3"] + FM + m["CaO"] - m["P2O5"]) * (NK + V)))
    return B, C


def log_eta(wt, T_C):
    """log10 eta (Pa s) for wt% comp dict (incl H2O, F) at T in degrees C."""
    m = to_molpct(wt)
    B, C = BC(m)
    T = T_C + 273.15
    return A + B / (T - C)


# --------------------------------------------------------------------------
# VALIDATION against paper Table 2 (iron-free andesite + 2.00 wt% H2O)
# Expected: B ~ 7720, C ~ 334, log eta(1273 K = 999.85 C) = 3.67
# --------------------------------------------------------------------------
def validate():
    andesite = dict(SiO2=62.40, TiO2=0.55, Al2O3=20.01, FeO=0.03, MnO=0.02,
                    MgO=3.22, CaO=9.08, Na2O=3.52, K2O=0.93, P2O5=0.12,
                    H2O=2.00, F=0.0)
    m = to_molpct(andesite)
    B, C = BC(m)
    le = A + B / (1273.0 - C)
    print("=== VALIDATION (Table 2) ===")
    print(f"  mol% SiO2={m['SiO2']:.2f} Al2O3={m['Al2O3']:.2f} H2O={m['H2O']:.2f}"
          f"  (paper: 62.38 / 11.79 / 6.80)")
    print(f"  B={B:.0f} (paper 7720)   C={C:.0f} (paper 334)")
    print(f"  log eta @1273 K = {le:.2f} (paper 3.67)")
    ok = abs(B - 7720) < 30 and abs(C - 334) < 5 and abs(le - 3.67) < 0.03
    print(f"  PASS={ok}\n")
    return ok


# --------------------------------------------------------------------------
# Monte Carlo propagation (triangular dists, n=10,000) -- P1 locked method
# --------------------------------------------------------------------------
def tri(mode, lo, hi, n):
    return np.random.triangular(lo, mode, hi, n)


def mc_logeta_vs_T(base_majors, H2O_tri, F_tri, T_grid_C, n=10000, seed=0):
    """base_majors: dict of anhydrous major wt% (SiO2..P2O5).
    H2O_tri/F_tri: (mode, lo, hi) wt% triangular params.
    Returns dict T_C -> (median, p2.5, p16, p84, p97.5) of log eta."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    H2O = tri(*H2O_tri, n)
    F = tri(*F_tri, n)
    out = {}
    for T_C in T_grid_C:
        le = np.empty(n)
        for i in range(n):
            wt = dict(base_majors); wt["H2O"] = H2O[i]; wt["F"] = F[i]
            le[i] = log_eta(wt, T_C)
        out[T_C] = (np.median(le), np.percentile(le, 2.5),
                    np.percentile(le, 16), np.percentile(le, 84),
                    np.percentile(le, 97.5))
    return out


# --------------------------------------------------------------------------
# PLACEHOLDER intermediate/high-K majors -- NOT an OT result.
# >>> AE: replace `ot_majors` with real Ok Tedi whole-rock / de-spiked ZHMI
#     majors (anhydrous, renormalized). H2O/F modes already locked. <<<
# Module-level so downstream tools (viscosity_path.py) can import it.
# --------------------------------------------------------------------------
ot_majors = dict(SiO2=63.0, TiO2=0.55, Al2O3=16.5, FeO=4.0, MnO=0.08,
                 MgO=2.2, CaO=4.2, Na2O=3.8, K2O=3.5, P2O5=0.20)  # PLACEHOLDER


if __name__ == "__main__":
    assert validate(), "Giordano model failed validation -- do not use."
    T_grid = [1000, 950, 900, 850, 800, 750]
    res = mc_logeta_vs_T(ot_majors,
                         H2O_tri=(4.0, 3.0, 5.0),   # mode 4 wt% (model input; non-unique)
                         F_tri=(0.45, 0.35, 0.55),  # mode 0.45 wt% (ZHAI fwd model)
                         T_grid_C=T_grid, n=10000)
    print("=== DEMO (PLACEHOLDER majors -- NOT an OT result) ===")
    print(" T(C)  median  [2.5  16  84  97.5]  log10 eta (Pa s)")
    for T_C in T_grid:
        med, p025, p16, p84, p975 = res[T_C]
        print(f" {T_C:4d}   {med:5.2f}   [{p025:.2f} {p16:.2f} {p84:.2f} {p975:.2f}]")
    print("\nSingle-point early-melt example (1000 C, mode inputs):")
    wt0 = dict(ot_majors); wt0["H2O"] = 4.0; wt0["F"] = 0.45
    print(f"  log10 eta = {log_eta(wt0, 1000):.2f} Pa s  (DEMO placeholder)")
