"""
Ok Tedi P1 -- melt-viscosity cooling path (two-branch) tool.

Reconstructs the evolution of melt log(eta) vs T for the EARLY Ok Tedi melt as it
cools, highlighting the slope change ACROSS H2O saturation (the "kink"). Output
feeds the companion manuscript's discussion, in the style of Li et al. (2024, Econ. Geol.
Fig. 17C/D). Built on the validated Giordano, Russell & Dingwell (2008, EPSL 271)
VFT model in `giordano2008.py`.

WHAT THIS IS / IS NOT
---------------------
- This is MELT viscosity (an intensive *property*) along a temperature path.
- It is NOT magma (suspension) viscosity and NOT a flux. No crystal/bubble
  multiplier is applied (Einstein-Roscoe / Costa 2005 / Mader 2013 -> P3).
- ISOBARIC assumption: above saturation H2O rises with crystallization; below
  saturation H2O is buffered CONSTANT at the solubility value, with excess water
  exsolving into a fluid phase (Lormand et al. 2024). We do NOT implement a
  decompression / degassing branch where dissolved H2O falls (that is P3).

H2O(T) COUPLING (SPEC section 2.2)
----------------------------------
- Undersaturated branch (T >= T_sat): H2O rises from H2O_initial to H2O_sat.
  Preferred: ingest a forward-model H2O(T) array (Lormand et al. 2024, already
  containing the Huber et al. 2010 b=2 cooling parameterization). Fallback when
  no array is supplied: monotone interpolation between (T_initial, H2O_initial)
  and (T_sat, H2O_sat), flagged as a PLACEHOLDER coupling.
- Saturated branch (T < T_sat): H2O = H2O_sat (constant). H2O is NEVER allowed
  to decrease post-saturation (asserted).

F retention: F stays in the melt across exsolution and is ~constant along the
path (Mercer et al. 2015).

COMPARISON PATH (constant H2O): alongside the enriching path we also compute a
second path over the SAME cooling temperatures but with H2O held CONSTANT at
~4 wt% (no enrichment). This isolates the purely thermal viscosity rise and
quantifies how much water build-up suppresses viscosity (the "enrichment
dividend" = constant - enriching). It carries its own triangular MC band.

CAVEAT (printed + written into figure/CSV headers): majors are held CONSTANT
along the path. Real residual-melt major evolution (becoming more silicic) would
make the post-saturation rise STEEPER, so this result is a LOWER BOUND.

References: Giordano, Russell & Dingwell (2008) EPSL 271, 123-134;
Huber et al. (2010); Lormand et al. (2024); Mercer et al. (2015); Li et al. (2024).
"""
import os
import sys
import argparse
import numpy as np

from giordano2008 import log_eta, tri, validate, ot_majors, mc_logeta_vs_T

def _resolve_out_dir():
    """Figure/CSV output dir. Generated files stay OUT of the git repo:
    default = the script-side `viscosity_path_out/` (git-ignored);
    `OT_FIG_DIR` overrides."""
    env = os.environ.get("OT_FIG_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "viscosity_path_out")


OUT_DIR = _resolve_out_dir()

# ----- locked P1 defaults (SPEC section 2.1) -----
# T_INITIAL is NOT a hydrous-granite liquidus: it is the onset of fractional
# crystallization of the mafic-to-intermediate PARENTAL magma at ~15 km
# (~1000 degC is well established for basaltic-to-andesitic arc magmas,
# Collins et al. 2021); the path covers the full differentiation interval.
# T_FINAL here extends BELOW the near-solidus end purely to display the
# saturated branch; the effective near-solidus T_final of the PMD end-member
# itself is ~790 degC (Ti-in-zircon: Schiller & Finger 2019; experiments:
# Zhang et al. 2020, Xiao et al. 2024, Grocolas et al. 2025).
T_INITIAL = 1000.0   # degC, undersaturated start (parental-magma FC onset)
T_FINAL = 650.0      # degC, plot end below saturation (NOT the PMD T_final ~790)
T_SAT = 750.0        # degC, saturation temperature (parameter; ideally forward-model coupled)
H2O_INITIAL = 4.0    # wt%, model input (NON-UNIQUE)
H2O_SAT = 10.0       # wt%, forward-model saturation value
F_WT = 0.45          # wt%, ZHAI forward model, ~constant along path (Mercer 2015)
DT = 5.0             # degC step


# --------------------------------------------------------------------------
# H2O(T) coupling
# --------------------------------------------------------------------------
def h2o_at_T(T_C, T_initial, T_sat, H2O_initial, H2O_sat, H2O_of_T=None):
    """Dissolved melt H2O (wt%) at temperature T_C along the cooling path.

    Undersaturated (T >= T_sat): ingest H2O_of_T if given (callable or
    (T_array, H2O_array) tuple), else monotone linear placeholder
    (T_initial,H2O_initial) -> (T_sat,H2O_sat).
    Saturated (T < T_sat): H2O = H2O_sat (constant, isobaric buffer).
    """
    if T_C < T_sat:
        return H2O_sat
    if H2O_of_T is not None:
        if callable(H2O_of_T):
            return float(H2O_of_T(T_C))
        Tarr, Harr = H2O_of_T
        # np.interp needs increasing x -> sort by T
        order = np.argsort(Tarr)
        return float(np.interp(T_C, np.asarray(Tarr)[order], np.asarray(Harr)[order]))
    # placeholder monotone interp on the undersaturated branch
    if T_initial == T_sat:
        return H2O_sat
    frac = (T_initial - T_C) / (T_initial - T_sat)  # 0 at start -> 1 at T_sat
    return H2O_initial + frac * (H2O_sat - H2O_initial)


# --------------------------------------------------------------------------
# Deterministic cooling path
# --------------------------------------------------------------------------
def cooling_path(majors, T_initial=T_INITIAL, T_final=T_FINAL, T_sat=T_SAT,
                 H2O_initial=H2O_INITIAL, H2O_sat=H2O_SAT, F=F_WT, dT=DT,
                 H2O_of_T=None):
    """Compute the two-branch cooling path.

    Returns a dict of equal-length numpy arrays:
      T_C, H2O_wt, logEta_path, logEta_cf, branch ('under'/'sat')
    plus 'coupling' = 'forward-model' or 'placeholder'.
    logEta_cf is the counterfactual with H2O FROZEN at H2O_initial (no
    enrichment): gap = logEta_cf - logEta_path quantifies the viscosity
    suppression "dividend" from water enrichment.
    """
    n = int(round((T_initial - T_final) / dT)) + 1
    T_grid = np.linspace(T_initial, T_final, n)
    coupling = "forward-model" if H2O_of_T is not None else "placeholder"

    H2O_wt = np.empty(n)
    logEta_path = np.empty(n)
    logEta_cf = np.empty(n)
    branch = np.empty(n, dtype=object)

    prev_h2o = None
    for i, T_C in enumerate(T_grid):
        h = h2o_at_T(T_C, T_initial, T_sat, H2O_initial, H2O_sat, H2O_of_T)
        # isobaric assertion: H2O must never decrease as we cool (no degassing branch)
        if prev_h2o is not None:
            assert h >= prev_h2o - 1e-9, (
                f"H2O decreased on cooling at T={T_C:.0f}C ({prev_h2o:.3f}->{h:.3f}); "
                "that is a decompression/degassing path (P3), not this isobaric path.")
        prev_h2o = h
        H2O_wt[i] = h
        branch[i] = "under" if T_C >= T_sat else "sat"

        wt = dict(majors); wt["H2O"] = h; wt["F"] = F
        logEta_path[i] = log_eta(wt, T_C)

        wt_cf = dict(majors); wt_cf["H2O"] = H2O_initial; wt_cf["F"] = F
        logEta_cf[i] = log_eta(wt_cf, T_C)

    return dict(T_C=T_grid, H2O_wt=H2O_wt, logEta_path=logEta_path,
                logEta_cf=logEta_cf, branch=branch, coupling=coupling,
                T_sat=T_sat)


def branch_slopes(path):
    """Linear-fit d(logEta)/dT on each branch. Returns dict with slopes
    (log per degC) and the kink ratio |sat slope| / |under slope|."""
    T = path["T_C"]; le = path["logEta_path"]; br = path["branch"]
    under = np.array([b == "under" for b in br])
    sat = ~under
    res = {}
    for name, mask in (("under", under), ("sat", sat)):
        if mask.sum() >= 2:
            slope = np.polyfit(T[mask], le[mask], 1)[0]
        else:
            slope = float("nan")
        res[name + "_slope"] = slope
    us = abs(res["under_slope"]) if res["under_slope"] == res["under_slope"] else float("nan")
    res["kink_ratio"] = (abs(res["sat_slope"]) / us) if us > 1e-12 else float("inf")
    return res


# --------------------------------------------------------------------------
# Monte Carlo along the coupled path (triangular, n=10,000) -- P1 locked method
# --------------------------------------------------------------------------
def monte_carlo_path(majors, T_initial=T_INITIAL, T_final=T_FINAL, T_sat=T_SAT,
                     dT=DT, F=F_WT, H2O_of_T=None,
                     H2O_init_tri=(4.0, 3.0, 5.0),    # mode 4 (non-unique)
                     H2O_sat_tri=(10.0, 8.5, 11.0),   # AE-TUNE range
                     F_tri=(0.45, 0.35, 0.55),
                     T_sat_tri=None,                  # e.g. (750, 720, 780); None = fixed
                     n=10000, seed=0):
    """Propagate uncertainty in H2O_initial, H2O_sat, F (and optionally T_sat)
    THROUGH the coupling. Returns (T_grid, percentiles_dict) where
    percentiles_dict maps stat -> array over T: 'median','p2.5','p16','p84','p97.5'.

    NB ranges marked AE-TUNE are placeholders to be vetted by the AE.
    """
    np.random.seed(seed)
    nT = int(round((T_initial - T_final) / dT)) + 1
    T_grid = np.linspace(T_initial, T_final, nT)

    H2O_i = tri(*H2O_init_tri, n)
    H2O_s = tri(*H2O_sat_tri, n)
    Fv = tri(*F_tri, n)
    if T_sat_tri is not None:
        Tsat_v = tri(*T_sat_tri, n)
    else:
        Tsat_v = np.full(n, T_sat)

    le = np.empty((nT, n))
    for j in range(n):
        for k, T_C in enumerate(T_grid):
            h = h2o_at_T(T_C, T_initial, Tsat_v[j], H2O_i[j], H2O_s[j], H2O_of_T)
            wt = dict(majors); wt["H2O"] = h; wt["F"] = Fv[j]
            le[k, j] = log_eta(wt, T_C)

    pct = dict(median=np.median(le, axis=1),
               p2_5=np.percentile(le, 2.5, axis=1),
               p16=np.percentile(le, 16, axis=1),
               p84=np.percentile(le, 84, axis=1),
               p97_5=np.percentile(le, 97.5, axis=1))
    return T_grid, pct


def monte_carlo_constant(majors, T_initial=T_INITIAL, T_final=T_FINAL, dT=DT,
                         H2O_tri=(4.0, 3.0, 5.0), F_tri=(0.45, 0.35, 0.55),
                         n=10000, seed=0):
    """COMPARISON path: H2O held CONSTANT (~4 wt%, NO enrichment) across the SAME
    cooling temperatures. Isolates the purely thermal viscosity rise -- i.e. what
    happens if water never builds up. This is the no-enrichment counterfactual,
    now propagated as a full MC band so it is directly comparable to the main
    coupled path. There is NO saturation kink here (H2O does not vary with T).
    Reuses the base `mc_logeta_vs_T` (constant-H2O triangular MC).

    Returns (T_grid, pct) with the same percentile-dict layout as
    `monte_carlo_path`.
    """
    nT = int(round((T_initial - T_final) / dT)) + 1
    T_grid = np.linspace(T_initial, T_final, nT)
    raw = mc_logeta_vs_T(majors, H2O_tri=H2O_tri, F_tri=F_tri,
                         T_grid_C=list(T_grid), n=n, seed=seed)
    med = np.array([raw[T][0] for T in T_grid])
    p25 = np.array([raw[T][1] for T in T_grid])
    p16 = np.array([raw[T][2] for T in T_grid])
    p84 = np.array([raw[T][3] for T in T_grid])
    p975 = np.array([raw[T][4] for T in T_grid])
    return T_grid, dict(median=med, p2_5=p25, p16=p16, p84=p84, p97_5=p975)


# --------------------------------------------------------------------------
# Melt-fraction -> temperature mapping (for overlaying apatite volatile curves)
# --------------------------------------------------------------------------
def map_F_to_T(F_melt, anchors):
    """Map a melt-fraction array F_melt onto temperature via monotone interp
    through (F, T) anchor points. `anchors` = list of (F, T) pairs, e.g.
        [(1.0, 1000), (F_sat, 750), (0.1, 650)]
    The apatite volatile model is ISOTHERMAL (computed at a single T_val); the
    F->T relation is an EXTERNAL petrological assumption (crystallization as the
    melt cools) that the USER supplies. Anchoring F_sat -> T_sat keeps the
    volatile saturation onset consistent with the viscosity path's kink.
    Returns T array (same length as F_melt)."""
    F_anch = np.array([a[0] for a in anchors], dtype=float)
    T_anch = np.array([a[1] for a in anchors], dtype=float)
    order = np.argsort(F_anch)              # np.interp needs increasing x (F)
    return np.interp(np.asarray(F_melt, dtype=float), F_anch[order], T_anch[order])


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
CAVEAT = ("majors held CONSTANT along path; real residual-melt evolution (more "
          "silicic) would steepen the post-saturation rise -> this is a LOWER BOUND. "
          "MELT viscosity only (no crystal/bubble loading); isobaric (no degassing).")


def write_csv(path, T_grid, pct, pct_const, fname):
    os.makedirs(OUT_DIR, exist_ok=True)
    fp = os.path.join(OUT_DIR, fname)
    # H2O_wt at mode inputs comes from the deterministic path on the same grid
    h2o = path["H2O_wt"]; cf = path["logEta_cf"]
    with open(fp, "w", encoding="utf-8") as f:
        f.write(f"# Ok Tedi P1 melt-viscosity cooling path. CAVEAT: {CAVEAT}\n")
        f.write(f"# coupling={path['coupling']} T_sat={path['T_sat']:.0f}C "
                "Giordano et al.(2008); H2O(T): Huber(2010)/Lormand(2024); F: Mercer(2015)\n")
        f.write("# 'const' columns = COMPARISON path with H2O held constant ~4 wt% (no "
                "enrichment) over the same T; logEta_counterfactual = its deterministic mode.\n")
        f.write("T_C,H2O_wt,logEta_median,p2.5,p16,p84,p97.5,"
                "logEta_const_median,const_p2.5,const_p16,const_p84,const_p97.5,"
                "logEta_counterfactual\n")
        for i, T_C in enumerate(T_grid):
            f.write(f"{T_C:.1f},{h2o[i]:.3f},{pct['median'][i]:.4f},"
                    f"{pct['p2_5'][i]:.4f},{pct['p16'][i]:.4f},{pct['p84'][i]:.4f},"
                    f"{pct['p97_5'][i]:.4f},"
                    f"{pct_const['median'][i]:.4f},{pct_const['p2_5'][i]:.4f},"
                    f"{pct_const['p16'][i]:.4f},{pct_const['p84'][i]:.4f},"
                    f"{pct_const['p97_5'][i]:.4f},{cf[i]:.4f}\n")
    return fp


def key_numbers(path, slopes, T_grid, pct):
    """Build the key-export string (printed + written to txt)."""
    T = path["T_C"]; le = path["logEta_path"]; cf = path["logEta_cf"]

    def at(Ttarget):
        idx = int(np.argmin(np.abs(T - Ttarget)))
        return le[idx], cf[idx]

    e_i, c_i = at(T[0])  # T_initial
    e_sat, c_sat = at(path["T_sat"])
    e_f, c_f = at(T[-1])  # T_final
    # constant-H2O (=4 wt%) comparison path is exactly logEta_cf; its slope is
    # purely thermal (no saturation kink).
    const_slope = np.polyfit(T, cf, 1)[0]
    lines = []
    # callers with real majors (e.g. ot_viscosity_volatile_overlay) override via
    # path["majors_note"] / path["caveat"]; the standalone script stays PLACEHOLDER.
    lines.append(f"=== KEY NUMBERS ({path.get('majors_note', 'PLACEHOLDER majors -- NOT an OT result')}) ===")
    lines.append(f"caveat: {path.get('caveat', CAVEAT)}")
    lines.append(f"coupling = {path['coupling']};  T_sat = {path['T_sat']:.0f} C")
    lines.append("-- enriching (coupled) path --")
    lines.append(f"log10 eta @ T_initial({T[0]:.0f}C)         = {e_i:.2f}")
    lines.append(f"log10 eta @ T_sat({path['T_sat']:.0f}C, flow window) = {e_sat:.2f}")
    lines.append(f"log10 eta @ T_final({T[-1]:.0f}C)          = {e_f:.2f}")
    lines.append(f"undersaturated-branch slope = {slopes['under_slope']:+.5f} log/degC")
    lines.append(f"saturated-branch slope      = {slopes['sat_slope']:+.5f} log/degC")
    lines.append(f"kink ratio |sat|/|under|    = {slopes['kink_ratio']:.2f}x")
    lines.append("-- constant-H2O ~4 wt% comparison path (no enrichment) --")
    lines.append(f"log10 eta @ T_initial({T[0]:.0f}C)         = {c_i:.2f}")
    lines.append(f"log10 eta @ {path['T_sat']:.0f}C                  = {c_sat:.2f}")
    lines.append(f"log10 eta @ T_final({T[-1]:.0f}C)          = {c_f:.2f}")
    lines.append(f"slope (purely thermal, no kink) = {const_slope:+.5f} log/degC")
    lines.append("-- enrichment dividend (constant - enriching) --")
    lines.append(f"gap @ T_initial = {c_i - e_i:+.2f}  (zero by construction; both H2O=4)")
    lines.append(f"gap @ T_sat     = {c_sat - e_sat:+.2f} log units")
    lines.append(f"gap @ T_final   = {c_f - e_f:+.2f} log units")
    lines.append(f"mean gap over path = {np.mean(cf - le):+.2f} log units")
    return "\n".join(lines)


def _draw_panel(ax, path, T_grid, pct, pct_const, xaxis=None, zone=None,
                iso_h2o=None, const_band=True, const_line=True,
                const_label=None, ylabel=True, iso_end_labels=True,
                ylim=None, xlim=None, band_layers=None, extra_lines=None):
    """Axes-level panel body shared by make_figure (single axes, optional
    volatile twin) and make_figure_panels (1x2 shared-y). Parameters mirror
    make_figure; additionally const_line=False drops the constant-H2O
    comparison path entirely, band AND median line (P1 melt-only figure,
    user decision 2026-08-08: the 3-6 wt% isopleths carry the reference,
    the dividend stays in key_numbers); ylabel / iso_end_labels let the
    panels variant label only the outer edges.

    ylim / xlim (optional, both None = autoscale as before): fixed axis limits
    for the manuscript figure. Given in DISPLAY order, so xlim carries its own
    direction -- xlim=(1.0, 0.0) or (1000, 750) already reads left->right and
    SUPPRESSES the xaxis['invert'] flip. ylim is applied before anything is
    drawn, because the saturation and zone labels anchor to get_ylim() and
    would otherwise be positioned against the autoscaled frame.

    band_layers (optional): REPLACES the two pct fill_between bands with a
    graded stack (user 2026-08-09). Ordered list of dicts {lo, hi (arrays on
    the x grid), color, label (optional; "_nolegend_" default)}, drawn in list
    order. Layers may NEST (draw outermost first, so inner layers paint on top)
    or TILE edge-to-edge, in which case order does not matter -- the P1 panels
    band tiles, one layer per member sorted by initial H2O. Solid colors
    sampled from a colormap, no alpha stacking. When given, pct/pct_const are
    ignored for fills (median lines unaffected).

    extra_lines (optional): additional reference trajectories on the SAME
    T grid (mapped through this panel's x coordinates). List of dicts
    {y, color, lw, ls, label, zorder}; drawn between the band (1.0) and the
    median path lines (2.2). Added for the dual-line ruling (2026-08-09) and
    kept as a generic hook when that ruling was reversed the same day (the
    orange best-fit line was dropped) -- so it currently has NO caller."""
    le = path["logEta_path"]
    br = path["branch"]; Tsat = path["T_sat"]
    under = np.array([b == "under" for b in br]); sat = ~under

    # horizontal-axis coordinates (default = temperature; xaxis overrides them)
    if xaxis is not None:
        xp = np.asarray(xaxis["path"]); xs = xaxis["sat"]
        xlabel = xaxis["label"]; xinvert = xaxis.get("invert", False)
        satlab = xaxis.get("sat_label", "sat")
    else:
        xp = np.asarray(T_grid); xs = Tsat
        xlabel = "Temperature (degC)"; xinvert = True
        satlab = f"T_sat={Tsat:.0f}C"

    # fixed y range BEFORE drawing: set_ylim also freezes y autoscale, and the
    # saturation / zone labels below anchor to get_ylim()
    if ylim is not None:
        ax.set_ylim(*ylim)

    # --- enriching (coupled) path: band, then median lines ---
    if band_layers is not None:
        # graded misfit-ranked envelopes (outermost/lightest first)
        for ly in band_layers:
            ax.fill_between(xp, ly["lo"], ly["hi"], color=ly["color"], lw=0,
                            zorder=1.0, label=ly.get("label", "_nolegend_"))
    else:
        # label fixed 2026-08-10: these fills were never Monte Carlo — they are
        # percentiles of the accepted multistart FAMILY (RMSE <= 5x best),
        # forwarded deterministically. The legacy "enriching MC" wording was
        # flagged 2026-08-09 as a leftover of the placeholder era.
        ax.fill_between(xp, pct["p2_5"], pct["p97_5"], color="#4c72b0",
                        alpha=0.15, label="enriching family 95%")
        ax.fill_between(xp, pct["p16"], pct["p84"], color="#4c72b0",
                        alpha=0.28, label="enriching family 68%")
    ax.plot(xp[under], le[under], "-", color="#1f3b73", lw=2.2,
            label="enriching path (undersat.)")
    ax.plot(xp[sat], le[sat], "--", color="#1f3b73", lw=2.2,
            label="enriching path (saturated)")
    if extra_lines:
        for el in extra_lines:
            ax.plot(xp, el["y"], color=el.get("color", "#D55E00"),
                    lw=el.get("lw", 1.4), ls=el.get("ls", "-"),
                    zorder=el.get("zorder", 1.8),
                    label=el.get("label", "_nolegend_"))
    # --- constant-H2O comparison path (optional): red band + median line ---
    if const_line:
        if const_band:
            ax.fill_between(xp, pct_const["p2_5"], pct_const["p97_5"],
                            color="#c44e52", alpha=0.12,
                            label="constant-H2O family 95%")
            ax.fill_between(xp, pct_const["p16"], pct_const["p84"],
                            color="#c44e52", alpha=0.22,
                            label="constant-H2O family 68%")
        ax.plot(xp, pct_const["median"], ":", color="#8c2d2f", lw=2.0,
                label=const_label or "constant H2O ~4 wt% (no enrichment)")
    # --- constant-H2O reference isopleths (same path, only H2O frozen) ---
    if iso_h2o is not None:
        vals = list(iso_h2o["values"])
        for k, (w, cv) in enumerate(zip(vals, iso_h2o["curves"])):
            # zorder between the MC fills (1) and the median lines (2)
            ax.plot(xp, cv, ls=(0, (4, 2.5)), color="0.45", lw=0.9, zorder=1.6,
                    label=(f"constant-H2O reference ({vals[0]:g}-{vals[-1]:g} wt%)"
                           if k == 0 else "_nolegend_"))
            if iso_end_labels:
                ax.annotate(f"{w:g} wt%", (xp[-1], cv[-1]), xytext=(4, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=6.5, color="0.35", annotation_clip=False)
    # saturation marker (neutral grey to avoid clashing with the comparison path)
    ax.axvline(xs, color="#555555", lw=1.0, ls="-.", alpha=0.8)
    ax.text(xs, ax.get_ylim()[1], f" {satlab}", color="#555555",
            va="top", ha="left", fontsize=8)
    if zone is not None:               # flagged engine-dependent x-interval (shaded)
        z0, z1 = sorted((float(zone["x0"]), float(zone["x1"])))
        ax.axvspan(z0, z1, color="0.5", alpha=0.10, zorder=0)
        ax.text((z0 + z1) / 2.0, ax.get_ylim()[0], zone.get("label", ""),
                va="bottom", ha="center", fontsize=6.5, color="0.35")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("log10 eta (Pa s)")
    if xlim is not None:
        ax.set_xlim(*xlim)     # display order already; must NOT also invert
    elif xinvert:
        ax.invert_xaxis()  # cooling left->right


def make_figure(path, T_grid, pct, pct_const, fname_stem, volatiles=None, xaxis=None,
                zone=None, iso_h2o=None, const_band=True, const_line=True,
                const_label=None, legend_out=None):
    """volatiles (optional): dict with arrays already mapped onto the T axis:
        T, H2O, Cl, F (all wt%); optional cl_scale/f_scale (default 10) and
        'source' label. When given, a right twin axis overlays melt H2O and
        Cl x cl_scale, F x f_scale (Cl/F magnified so they are visible; the
        magnification is annotated in the legend).

    xaxis (optional): re-parameterise the horizontal axis away from temperature
        (everything else unchanged). Dict with:
          path : x values for the viscosity series (same length as T_grid)
          vol  : x values for the volatile series (same length as volatiles['T'])
          sat  : x value of the saturation marker
          label, sat_label, invert (bool)  -- axis label / vline text / direction.

    zone (optional): shade an x-interval where the plotted quantity is flagged
        engine-/model-dependent (e.g. the F<0.63 majors zone, ruling 2026-07-08).
        Dict with x0, x1 (interval, either order) and label (small in-zone text).

    iso_h2o (optional): constant-H2O reference isopleths. Dict with
          values : H2O wt% list (e.g. [3,4,5,6])
          curves : matching list of logEta arrays on the same grid as `path`
        Each curve must be computed on the SAME path (evolving majors(F),
        halogen F(T), same T grid) with only H2O frozen, so the spacing to the
        main curve reads as the pure H2O effect. Thin grey dashed lines,
        wt% labelled at the cooling end, one shared legend entry.

    const_band / const_line / const_label: draw the constant-H2O comparison MC
        band (default True) / draw the comparison path at all (default True;
        const_line=False drops band AND median line -- P1 melt-only figure,
        user decision 2026-08-08, the isopleths carry the reference) /
        override that path's legend text (default keeps the standalone
        "~4 wt%" wording). const_band=False keeps only the median line.

    legend_out: force the legend outside the axes (right). Default None = auto
        (outside only with the volatile twin axis, as before). True frees the
        in-axes top strip for the T_sat label and keeps the legend off the
        isopleths (P1 melt-only figure, decision 2026-08-08)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    # legend placement: outside right with the volatile twin axis (as before),
    # or when forced via legend_out; widen the canvas so the axes keep their
    # proportions once the legend claims the right margin
    legend_outside = legend_out if legend_out is not None else volatiles is not None
    if volatiles:
        figsize = (8.8, 4.8)
    elif legend_outside:
        figsize = (7.6, 4.6)
    else:
        figsize = (6.0, 4.6)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_panel(ax, path, T_grid, pct, pct_const, xaxis=xaxis, zone=zone,
                iso_h2o=iso_h2o, const_band=const_band, const_line=const_line,
                const_label=const_label)

    # --- optional right twin axis: melt F-Cl-H2O volatile trajectory ---
    ax2 = None
    if volatiles is not None:
        cls = volatiles.get("cl_scale", 10.0)
        fsc = volatiles.get("f_scale", 10.0)
        xv = np.asarray(xaxis["vol"]) if xaxis is not None else np.asarray(volatiles["T"])
        ax2 = ax.twinx()
        # Wong colour-blind-safe palette (matches ot_test plot_OT_melt_evolution.py)
        ax2.plot(xv, volatiles["H2O"], "-", color="#0072B2", lw=1.6, marker="o",
                 ms=3, mfc="white", mew=0.6, label="melt H2O")
        ax2.plot(xv, np.asarray(volatiles["Cl"]) * cls, "--", color="#D55E00", lw=1.6,
                 marker="s", ms=3, mfc="white", mew=0.6, label=f"melt Cl x{cls:g}")
        ax2.plot(xv, np.asarray(volatiles["F"]) * fsc, "-.", color="#009E73", lw=1.6,
                 marker="^", ms=3, mfc="white", mew=0.6, label=f"melt F x{fsc:g}")
        ax2.set_ylabel(f"melt H2O (wt%);  Cl x{cls:g}, F x{fsc:g} (wt%)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_ylim(bottom=0)

    mtag = path.get("majors_tag", "PLACEHOLDER majors")
    ttl = f"Ok Tedi early-melt viscosity + melt volatiles ({mtag})" if volatiles \
        else f"Ok Tedi early-melt viscosity cooling path ({mtag})"
    ax.set_title(ttl, fontsize=9)
    # combined legend (viscosity axis + volatile axis)
    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    if legend_outside:
        # outside right: clears the curves, the T_sat top label and the isopleth
        # end labels; anchor further out when the volatile twin axis occupies
        # the right margin with its own tick/axis labels
        ax.legend(h1, l1, fontsize=6.8, loc="upper left",
                  bbox_to_anchor=(1.14 if ax2 is not None else 1.08, 1.0),
                  borderaxespad=0.0)
    else:
        ax.legend(h1, l1, fontsize=7, loc="best")
    # callers with real majors override the standalone-script default via
    # path["fig_note"] (e.g. the OT LLD coupled runs, where majors are NOT constant)
    note = path.get("fig_note",
                    "LOWER BOUND: majors constant; melt (not magma) viscosity; isobaric.")
    if volatiles is not None and volatiles.get("source"):
        note += f"  Volatiles: {volatiles['source']}."
    # wrap: a single long line + bbox_inches="tight" would blow the canvas up
    # horizontally and squeeze the axes to a sliver
    import textwrap
    note = "\n".join(textwrap.wrap(note, 95))
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color="#555")
    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, fname_stem + ".pdf")
    png = os.path.join(OUT_DIR, fname_stem + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def make_figure_panels(path, T_grid, pct, pct_const, fname_stem, panels,
                       iso_h2o=None, const_band=False, const_line=False,
                       const_label=None, ylim=None, band_layers=None,
                       band_cbar=None, fig_note=None, extra_lines=None):
    """1x2 shared-y variant of make_figure -- the P1 melt-only main figure
    (user decisions 2026-08-08): panel (a) melt-fraction axis LEFT, panel (b)
    temperature axis RIGHT, panel letters, ONE legend outside right, no
    volatile twin axis. `panels` is an ordered list of dicts
    {"xaxis": make_figure-style dict or None (= temperature axis),
    "zone": ..., "letter": "(a)", "xlim": (left, right) or absent}. Isopleth
    wt% end labels go on the LAST (outer) panel only, so they never collide
    with the neighbouring panel. `ylim` (shared, sharey=True) and the per-panel
    `xlim` fix the manuscript frame; both default to autoscale.

    band_layers / band_cbar (user 2026-08-09): graded misfit-ranked band.
    band_layers is passed straight to _draw_panel (see there); band_cbar
    (optional) draws a small vertical colorbar under the outside-right legend
    -- dict {colors (innermost first), bounds (len+1, e.g. rank-%% edges),
    ticks, label}. fig_note (optional) overrides path['fig_note'] for THIS
    figure only, so the shared path dict can keep serving callers whose band
    definition differs (the coupled overlay keeps the pct band + pin note).
    Replaces the former per-axis single files (vsT/vsF). Written as PDF +
    PNG + SVG (SVG added 2026-08-08 for the manuscript vector submission);
    returns (pdf, png, svg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axs = plt.subplots(1, len(panels), figsize=(11.8, 4.6), sharey=True)
    axs = np.atleast_1d(axs)
    for i, (ax, pn) in enumerate(zip(axs, panels)):
        _draw_panel(ax, path, T_grid, pct, pct_const, xaxis=pn.get("xaxis"),
                    zone=pn.get("zone"), iso_h2o=iso_h2o,
                    const_band=const_band, const_line=const_line,
                    const_label=const_label, ylabel=(i == 0),
                    iso_end_labels=(i == len(panels) - 1),
                    ylim=ylim, xlim=pn.get("xlim"), band_layers=band_layers,
                    extra_lines=extra_lines)
        ax.text(0.015, 0.985, pn.get("letter", f"({chr(97 + i)})"),
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                fontweight="bold")
    mtag = path.get("majors_tag", "PLACEHOLDER majors")
    fig.suptitle(f"Ok Tedi early-melt viscosity cooling path ({mtag})",
                 fontsize=9)
    h1, l1 = axs[-1].get_legend_handles_labels()
    axs[-1].legend(h1, l1, fontsize=6.8, loc="upper left",
                   bbox_to_anchor=(1.10, 1.0), borderaxespad=0.0)
    note = fig_note if fig_note is not None else path.get(
        "fig_note",
        "LOWER BOUND: majors constant; melt (not magma) viscosity; isobaric.")
    note = "\n".join(textwrap.wrap(note, 130))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if band_cbar is not None:
        # small vertical colorbar for the graded band, outside right BELOW the
        # legend (same margin column; bbox_inches="tight" picks it up)
        from matplotlib import cm
        from matplotlib.colors import BoundaryNorm, ListedColormap
        pos = axs[-1].get_position()
        cax = fig.add_axes([pos.x1 + 0.035, pos.y0 + 0.05,
                            0.013, 0.42 * (pos.y1 - pos.y0)])
        cmap = ListedColormap(list(band_cbar["colors"]))
        norm = BoundaryNorm(list(band_cbar["bounds"]), cmap.N)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          ticks=band_cbar.get("ticks"))
        cb.ax.tick_params(labelsize=6)
        cb.set_label(band_cbar.get("label", ""), fontsize=6.5)
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=6.5,
             color="#555")
    pdf = os.path.join(OUT_DIR, fname_stem + ".pdf")
    png = os.path.join(OUT_DIR, fname_stem + ".png")
    svg = os.path.join(OUT_DIR, fname_stem + ".svg")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return pdf, png, svg


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Ok Tedi P1 melt-viscosity cooling path")
    ap.add_argument("--validate", action="store_true",
                    help="run Table 2 self-check and exit")
    ap.add_argument("--n", type=int, default=10000, help="Monte Carlo sample size")
    ap.add_argument("--dt", type=float, default=DT, help="temperature step (degC)")
    ap.add_argument("--no-fig", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    if not validate():
        sys.exit("Giordano model failed Table 2 validation -- refusing to run.")
    if args.validate:
        print("validate: PASS")
        return

    print("=" * 68)
    print(" PLACEHOLDER majors -- NOT an OT result.")
    print(" >>> AE: replace `ot_majors` (giordano2008.py) with real OT whole-rock /")
    print("     de-spiked ZHMI majors; optionally pass forward-model H2O(T). <<<")
    print("=" * 68)

    majors = dict(ot_majors)  # PLACEHOLDER
    path = cooling_path(majors, dT=args.dt)
    slopes = branch_slopes(path)
    T_grid, pct = monte_carlo_path(majors, dT=args.dt, n=args.n)
    # comparison path: H2O held constant ~4 wt% over the SAME cooling temps
    T_grid_c, pct_const = monte_carlo_constant(majors, dT=args.dt, n=args.n)

    csv = write_csv(path, T_grid, pct, pct_const, "ot_viscosity_path_PLACEHOLDER.csv")
    kn = key_numbers(path, slopes, T_grid, pct)
    print("\n" + kn)
    os.makedirs(OUT_DIR, exist_ok=True)
    txt = os.path.join(OUT_DIR, "ot_viscosity_path_PLACEHOLDER_keynums.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(kn + "\n")

    print(f"\nwrote: {csv}")
    print(f"wrote: {txt}")
    if not args.no_fig:
        pdf, png = make_figure(path, T_grid, pct, pct_const,
                               "ot_viscosity_path_PLACEHOLDER")
        print(f"wrote: {pdf}")
        print(f"wrote: {png}")


if __name__ == "__main__":
    main()
