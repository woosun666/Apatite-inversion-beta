#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ok Tedi P1 -- interactive GUI for the Lormand apatite Cl-OH-F workflow
======================================================================
The Lormand original has NO widget GUI: its "interface" is edit-params-in-
script -> run 'Explore' -> look at figure 2 (target trend vs observed ratios)
-> iterate by eye. This GUI turns that loop into a live one:

  TAB 1  Eye-fit    : 9 params + K_spec sliders -> target curve redrawn live
                      over the observed apatite ratios (matrix vs ZIR colours,
                      saturation-onset star). LINEAR axes (user convention):
                      XF/XOH vs XCl/XOH  and  XF/XCl vs XCl/XOH.
  TAB 2  Volatiles  : melt H2O / Cl / F wt% vs melt fraction F for the current
                      params (live plot_OT_melt_evolution).
  TAB 3  Viscosity  : GRD08 coupled log eta vs T preview driven by the current
                      volatile trajectory (needs a T-F curve; majors are the
                      PLACEHOLDER ot_majors -- clearly labelled).
  TAB 4  Multistart : set BOUNDS + starts, run the multistart in a background
                      process pool (live progress), then view the accepted-
                      family parameter table, RMSE histogram and curve bundle;
                      export the MATLAB-format Multioutput CSV.

Everything computational is REUSED from the validated engine -- apatite_model
(forward physics), apatite_inversion (fixed-point forward, multistart workers),
giordano2008 (GRD08) -- nothing is re-implemented here.

Run:  python apatite_gui.py
Env (all optional): OT_OBS (observed ratios csv), OT_TF (T-F curve csv),
                    OT_OUT (default output dir).
Deps: stdlib tkinter + numpy/scipy/pandas/matplotlib (already required).
"""
from __future__ import annotations

import csv
import io
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---- application name -------------------------------------------------------
# Single source for the name shown in the window title. When frozen, PyInstaller
# names the executable from its --name flag, so read it back from there: a build
# packaged under a different name (e.g. apa-inversion-beta2) then labels itself
# correctly without editing this file. Script runs fall back to the literal.
APP_NAME = (os.path.splitext(os.path.basename(sys.executable))[0]
            if getattr(sys, "frozen", False) else "apa-inversion-beta")


# ---- safe export base -------------------------------------------------------
# 2026-08-08 data-loss lesson (Melt S): in a frozen onefile app anything
# written under the OS temp dir is invisible at best (otproj_* project
# extractions) and deleted with the app at worst (the _MEI bootloader dir).
# Every default export base below must therefore refuse temp-rooted paths.
def _temp_rooted(path):
    """True if path is inside the OS temp dir (case-folded, sep-safe)."""
    try:
        p = os.path.normcase(os.path.abspath(path))
        t = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
    except (OSError, ValueError):
        return False
    return p == t or p.startswith(t.rstrip(os.sep) + os.sep)


def visible_out_base():
    """Default base dir for user-facing exports: OT_OUT > cwd >
    ~\\<app>-output, skipping anything rooted in the OS temp dir (a frozen
    onefile app can be launched with a TEMP cwd)."""
    for c in (os.environ.get("OT_OUT"), os.getcwd()):
        if c and not _temp_rooted(c):
            return c
    return os.path.join(os.path.expanduser("~"), APP_NAME + "-output")

# ---- splash screen ----------------------------------------------------------
# Shown BEFORE the heavy imports below (numpy/engine/matplotlib take 1-2 s):
# that is where the perceived startup delay lives, so the splash must go up
# first. Script-run only (no side effects when this module is imported).
# Drop a `splash.png` next to this file for a custom image (PNG/GIF, Tk 8.6;
# splash.png is version-controlled despite the *.png ignore rule -- see the
# !splash.png re-include in .gitignore); otherwise a text card shows.
_SPLASH = _SPLASH_ROOT = None


def _make_splash():
    root = tk.Tk()
    root.withdraw()                      # main window stays hidden until built
    sp = tk.Toplevel(root)
    sp.overrideredirect(True)            # borderless card
    img = None
    # PyInstaller onefile extracts bundled data to sys._MEIPASS, which is NOT the
    # directory of __file__; onedir and a plain script run both resolve correctly
    # from __file__. Check _MEIPASS first so one code path serves all three.
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "splash.png")
    if os.path.exists(p):
        try:
            img = tk.PhotoImage(file=p)
        except tk.TclError:
            img = None
    if img is not None:
        lbl = tk.Label(sp, image=img, bd=1, relief="solid")
        lbl.image = img                  # keep a reference
        w, h = img.width(), img.height()
    else:
        lbl = tk.Label(sp, text="OT apatite Cl-OH-F explorer\n\nloading …",
                       font=("Segoe UI", 13), width=36, height=7,
                       bg="#1d3557", fg="white", bd=1, relief="solid")
        w, h = 400, 170
    lbl.pack()
    sw, sh = sp.winfo_screenwidth(), sp.winfo_screenheight()
    sp.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
    sp.attributes("-topmost", True)
    sp.update()                          # paint NOW, before imports block Tk
    return root, sp


if __name__ == "__main__" and "--multiprocessing-fork" not in sys.argv:
    # NOT in multiprocessing children: frozen (PyInstaller) workers re-run this
    # module as __main__ before freeze_support() can intercept — without the
    # argv guard every multistart worker would pop its own splash.
    _SPLASH_ROOT, _SPLASH = _make_splash()

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apatite_model as am              # noqa: E402
import apatite_inversion as inv         # noqa: E402
from giordano2008 import log_eta, ot_majors, validate as g08_validate  # noqa: E402

import matplotlib                       # noqa: E402
matplotlib.use("TkAgg")
# v2 B3: ONE style for every embedded figure (five tabs + popout windows).
# Display-only: exported figures inherit the same look.
matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"],
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.9,
})
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure    # noqa: E402

# ---------------------------------------------------------------------------
# Parameter metadata: order, slider range, display resolution
# ---------------------------------------------------------------------------
PARAM_ORDER = inv.PARAM_ORDER           # 9 fitted params (engine order)
# 26.6.24 multi-target envelope: bracketing-target colours (ot_target_envelope.py)
ENV_COL = {"low": "#0072B2", "central": "#000000", "high": "#D55E00"}
# 26.6.24 LOW/HIGH suggestion factors for (F_init, Cl_init) relative to central
ENV_SCALE = {"low": (0.72, 0.80), "high": (1.30, 1.30)}
PARAM_RANGE = {                         # generous slider ranges (NOT fit bounds)
    "H2O_init":   (1.0, 8.0),
    "Cl_init":    (0.01, 1.0),
    "F_init":     (0.01, 1.0),
    "D_xl_m_Cl":  (0.0, 0.6),
    "D_xl_m_F":   (0.0, 2.5),
    "D_xl_m_H2O": (0.0, 0.5),
    "D_fl_m_Cl":  (5.0, 50.0),
    "D_fl_m_F":   (0.5, 10.0),
    "H2O_Sat":    (4.0, 12.0),
}
KSPEC_RANGE = (0.30, 0.60)
NUDGE_DIV = 20.0                        # +/- button step = slider range / 20
# colormap choices for the multistart curve fan (2026-08-04); "jet" stays the
# default = the Lormand-MATLAB look users know
MS_CMAPS = ("jet", "turbo", "viridis", "plasma", "coolwarm", "Greys")

# start-up params = the OT-test-26.6.25 run-log target (current working set)
DEFAULT_PARAMS = dict(
    H2O_init=4.0, Cl_init=0.51, F_init=0.46,
    D_xl_m_Cl=0.05, D_xl_m_F=1.5, D_xl_m_H2O=0.12,
    D_fl_m_Cl=22.0, D_fl_m_F=2.5, H2O_Sat=10.0,
)
DEFAULT_KSPEC = 0.463
NUM_F_LIVE = 120                        # live forward resolution (fast)
NUM_F_FIT = 100                         # multistart forward resolution (engine default)

WONG = dict(blue="#0072B2", orange="#D55E00", green="#009E73",
            purple="#CC79A7", grey="#666666")
OVERLAY_COLORS = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#666666"]

# parameter grouping for the collapsible left-panel sections (task 1.1)
PARAM_GROUPS = [
    ("Initial melt composition", ["H2O_init", "Cl_init", "F_init"]),
    ("Crystal/melt partition coefficients",
     ["D_xl_m_Cl", "D_xl_m_F", "D_xl_m_H2O"]),
    ("Fluid/melt partition coefficients", ["D_fl_m_Cl", "D_fl_m_F"]),
    ("Model parameters", ["H2O_Sat", "K_spec"]),
]

# tooltip text per parameter (task 1.2) -- edit the "ref" entries freely;
# desc/range mirror GUI_MANUAL.md section 5
PARAM_INFO = {
    "H2O_init":   dict(desc="initial melt H2O, wt%",
                       range="typical 3-5", ref="Lormand et al. 2024"),
    "Cl_init":    dict(desc="initial melt Cl, wt%",
                       range="typical 0.2-0.6", ref="Lormand et al. 2024"),
    "F_init":     dict(desc="initial melt F, wt%",
                       range="typical 0.1-0.5", ref="Lormand et al. 2024"),
    "D_xl_m_Cl":  dict(desc="crystal/melt partition coefficient, Cl",
                       range="typical 0.02-0.4", ref="Lormand et al. 2024"),
    "D_xl_m_F":   dict(desc="crystal/melt partition coefficient, F "
                            "(>1 = compatible)",
                       range="typical 0.5-2", ref="Lormand et al. 2024"),
    "D_xl_m_H2O": dict(desc="crystal/melt partition coefficient, H2O",
                       range="typical ~0.12", ref="Lormand et al. 2024"),
    "D_fl_m_Cl":  dict(desc="fluid/melt Cl partition coefficient "
                            "(drives the Cl strip after saturation)",
                       range="typical 15-30", ref="Lormand et al. 2024"),
    "D_fl_m_F":   dict(desc="fluid/melt F partition coefficient",
                       range="typical 1.5-5", ref="Lormand et al. 2024"),
    "H2O_Sat":    dict(desc="H2O saturation (solubility) value, wt% -- "
                            "sets the exsolution onset",
                       range="typical 8-11", ref="Lormand et al. 2024"),
    "K_spec":     dict(desc="OH-H2O speciation constant",
                       range="typical 0.36-0.47", ref="Humphreys et al. 2021"),
    # T-F formula section (Huber et al. 2010 cooling parameterization).
    # NB T_initial / T_final are the EFFECTIVE bounds of the modeled
    # crystallization interval, not a granite liquidus / solidus.
    "T_initial":  dict(desc="effective START of the modeled crystallization "
                            "interval, degC. NOT the liquidus of a hydrous "
                            "granitic melt: onset of fractional "
                            "crystallization of a mafic-to-intermediate "
                            "parental magma at ~15 km, which differentiates "
                            "toward the felsic melts of porphyry Cu "
                            "reservoirs",
                       range="~1000 for basaltic-to-andesitic arc magmas",
                       ref="Huber et al. 2010; Collins et al. 2021"),
    "T_final":    dict(desc="effective END of the modeled interval "
                            "(near-solidus), degC; ~790 for the PMD "
                            "end-member per Ti-in-zircon thermometry and "
                            "recent experimental constraints",
                       range="typical 750-850 for evolved porphyry systems",
                       ref="Schiller & Finger 2019; Zhang et al. 2020; "
                           "Xiao et al. 2024; Grocolas et al. 2025"),
    "b":          dict(desc="cooling coefficient of the Huber law "
                            "F = ((T-T_final)/(T_initial-T_final))^b; "
                            "b=1 linear, b=2 = the B2 curve of the OT "
                            "26.6.25 run",
                       range="typical 1-2", ref="Huber et al. 2010"),
}


class Tooltip:
    """Hover tooltip on a widget (500 ms delay, plain Toplevel)."""

    def __init__(self, widget, text):
        self.widget, self.text = widget, text
        self._tip = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.widget.after(500, self._show)

    def _cancel(self):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", relief="solid",
                 borderwidth=1, background="#ffffe0",
                 font=("TkDefaultFont", 8), wraplength=320,
                 padx=4, pady=2).pack()

    def _hide(self, _e=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def ui_font(name, **overrides):
    """Named font DERIVED from the real TkDefaultFont (v2 B2/B4), created on
    first use (needs a Tk root). The old ("TkDefaultFont", 9, "bold") tuples
    passed a named font as a font FAMILY — Windows can't resolve that family
    and silently fell back to a serif/monospace look that clashed with the
    YaHei/Segoe UI everywhere else."""
    from tkinter import font as tkfont
    try:
        return tkfont.nametofont(name)
    except tk.TclError:
        base = tkfont.nametofont("TkDefaultFont")
        f = tkfont.Font(name=name, **base.actual())
        f.configure(**overrides)
        return f


def section_font():
    return ui_font("OTSectionFont", weight="bold")


def small_font():
    return ui_font("OTSmallFont", size=8)


class CollapsibleFrame(ttk.Frame):
    """A LabelFrame-like section with a clickable header that collapses the
    body. Widgets go into `.body`; state via .collapsed / set_collapsed()."""

    def __init__(self, parent, title):
        super().__init__(parent)
        self.title = title
        self._open = True
        self._hdr = ttk.Label(self, text="v " + title, cursor="hand2",
                              font=section_font())
        self._hdr.pack(fill="x", pady=(2, 0))
        self._hdr.bind("<Button-1>", lambda *_: self.toggle())
        self.body = ttk.Frame(self, padding=(8, 2, 0, 2))
        self.body.pack(fill="x")

    @property
    def collapsed(self):
        return not self._open

    def toggle(self):
        self.set_collapsed(self._open)

    def set_collapsed(self, collapsed):
        self._open = not collapsed
        self._hdr.config(text=("> " if collapsed else "v ") + self.title)
        if collapsed:
            self.body.forget()
        else:
            self.body.pack(fill="x")

# ---- project files (.otproj = zip: manifest.json + embedded data + run CSVs) ----
PROJ_EXT = ".otproj"
PROJ_VERSION = 1
AUTOSAVE_MS = 5 * 60 * 1000             # M0-4 crash-recovery autosave period
RECENT_FILE = os.path.join(os.path.expanduser("~"), ".ot_apatite_gui_recent.json")
MAX_RECENT = 8


def load_recent():
    try:
        with open(RECENT_FILE, encoding="utf-8") as fh:
            return [p for p in json.load(fh) if os.path.exists(p)]
    except Exception:
        return []


def push_recent(path):
    lst = [path] + [p for p in load_recent() if p != path]
    try:
        with open(RECENT_FILE, "w", encoding="utf-8") as fh:
            json.dump(lst[:MAX_RECENT], fh)
    except Exception:
        pass


TIMING_FILE = os.path.join(os.path.expanduser("~"), ".ot_apatite_gui_timing.json")
# v2 A2: window geometry lives in a LOCAL per-user file (machine-specific —
# deliberately NOT in the .otproj, which travels between machines)
UI_STATE_FILE = os.path.join(os.path.expanduser("~"), ".ot_apatite_gui_ui.json")


def load_ui_state():
    try:
        with open(UI_STATE_FILE, encoding="utf-8") as fh:
            return dict(json.load(fh))
    except Exception:
        return {}


def save_ui_state(**kw):
    st = load_ui_state()
    st.update(kw)
    try:
        with open(UI_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
    except Exception:
        pass


def fmt_dur(sec):
    sec = max(0, int(round(sec)))
    return f"{sec//60}m {sec%60:02d}s" if sec >= 60 else f"{sec}s"


def load_fit_time():
    """Per-start wall time x workers (CPU-seconds/start) from previous runs
    on this machine; None if never measured."""
    try:
        with open(TIMING_FILE, encoding="utf-8") as fh:
            return float(json.load(fh)["per_start_cpu_s"])
    except Exception:
        return None


def save_fit_time(per_start_cpu_s):
    try:
        with open(TIMING_FILE, "w", encoding="utf-8") as fh:
            json.dump({"per_start_cpu_s": per_start_cpu_s}, fh)
    except Exception:
        pass


def results_to_csv_text(results):
    """Multistart results -> MATLAB Multioutput-format CSV text."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(inv.MO_COLS)
    for rmse, x in results:
        w.writerow(["%.12g" % float(xi) for xi in x] + ["%.12g" % rmse])
    return buf.getvalue()


def results_from_csv_text(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        x = [float(r[c]) for c in inv.MO_COLS[:-1]]
        out.append((float(r["RMSE"]), np.array(x)))
    return out


def make_icon(size=32):
    """Programmatic window icon (task 5.2): a stylized apatite hexagon
    (c-axis section), teal fill with a darker rim, transparent corners."""
    import math
    img = tk.PhotoImage(width=size, height=size)
    c = (size - 1) / 2.0

    def hexagon(r):
        return [(c + r * math.cos(math.radians(a + 30)),
                 c + r * math.sin(math.radians(a + 30)))
                for a in range(0, 360, 60)]

    def inside(pts, x, y):
        sign = None
        for i in range(6):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 6]
            cr = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if cr == 0:
                continue
            if sign is None:
                sign = cr > 0
            elif (cr > 0) != sign:
                return False
        return True

    outer, inner = hexagon(size * 0.44), hexagon(size * 0.33)
    for yy in range(size):
        for xx in range(size):
            if inside(inner, xx, yy):
                img.put("#3aa9a0", (xx, yy))       # apatite teal
            elif inside(outer, xx, yy):
                img.put("#1f6e6b", (xx, yy))       # rim
    return img


# ---------------------------------------------------------------------------
# Pure computation helpers (no Tk -- unit-testable)
# ---------------------------------------------------------------------------
def full_params(p9):
    p = dict(p9)
    p["D_fl_m_PrimBoilH2O"] = 0.0
    return p


def compute_state(p9, k_spec, F, T_K):
    """One forward run -> everything the tabs need. Returns dict or raises."""
    p = full_params(p9)
    cloh, foh, conv = inv.forward_ratios(p, F, T_K, k_spec, with_conv=True)
    vol = am.volatile_evolution(F, p)
    fim = vol["Fluid_inc_mass"]
    sat_i = int(np.argmax(fim > 0)) if (fim > 0).any() else None
    return dict(
        F=F, T_K=T_K, cloh=cloh, foh=foh, fcl=foh / cloh, conv=conv,
        H2O=vol["H2O_rem"] * 100.0, Cl=vol["Cl_rem"] * 100.0,
        Fwt=vol["F_rem"] * 100.0, sat_i=sat_i,
    )


def load_tf_curve(path):
    """(F, T_K, T_of_F) from a Var1=T_C / Var2=F csv (same rules as the engine)."""
    F, T_K = am.build_F_and_T(path, None, NUM_F_LIVE)

    def T_of_F(fv):
        order = np.argsort(F)
        return np.interp(np.asarray(fv, float), F[order], (T_K - 273.15)[order])
    return F, T_K, T_of_F


def formula_tf_curve(T_initial, T_final, b, n=NUM_F_LIVE):
    """(F, T_K, T_of_F) from the Huber et al. (2010) parameterization
    F = ((T-T_final)/(T_initial-T_final))^b -- same tuple contract as
    load_tf_curve; T_of_F is the analytic inverse (returns degC)."""
    F, T_K = am.generate_tf_curve(T_initial, T_final, b, n)
    Ti, Tf, bb = float(T_initial), float(T_final), float(b)

    def T_of_F(fv):
        fv = np.clip(np.asarray(fv, float), 0.0, None)
        return Tf + (Ti - Tf) * fv ** (1.0 / bb)
    return F, T_K, T_of_F


def load_observed(path):
    """Flexible observed-ratio loader: accepts the test_OT.csv layout
    (F/OH, Cl/OH, F/Cl + 'Rock type', ZIR flag in the value) or the generic
    Cl_OH / F_OH / F_Cl headers of apatite_inversion.load_observed_ratios."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names):
        for nm in names:
            if nm.lower() in cols:
                return cols[nm.lower()]
        return None

    c_cloh = pick("Cl/OH", "Cl_OH", "XCl_div_XOH")
    c_foh = pick("F/OH", "F_OH", "XF_div_XOH")
    c_fcl = pick("F/Cl", "F_Cl", "XF_div_XCl")
    if not (c_cloh and c_foh):
        raise ValueError("need Cl/OH + F/OH columns (or Cl_OH/F_OH)")
    c_grp = pick("Rock type", "group", "ZHAI_matrix", "type")
    out = dict(
        cloh=pd.to_numeric(df[c_cloh], errors="coerce").to_numpy(),
        foh=pd.to_numeric(df[c_foh], errors="coerce").to_numpy(),
        fcl=(pd.to_numeric(df[c_fcl], errors="coerce").to_numpy()
             if c_fcl else None),
    )
    ok = np.isfinite(out["cloh"]) & np.isfinite(out["foh"])
    if c_grp is not None:
        # fillna first: newer pandas keeps NaN through astype(str), which made
        # blank-group rows survive the filter (pre-refactor bug)
        grp = df[c_grp].fillna("").astype(str)
        ok &= (grp.str.strip().ne("") & grp.str.lower().ne("nan")).to_numpy()
        out["is_zir"] = grp.str.contains("ZIR", case=False, na=False).to_numpy()[ok]
    else:
        out["is_zir"] = np.zeros(int(ok.sum()), bool)
    for k in ("cloh", "foh", "fcl"):
        if out[k] is not None:
            out[k] = out[k][ok]
    out["raw"] = df.loc[ok].reset_index(drop=True)   # source rows (task 2.4)
    return out


def zir_misfit(curve_x, curve_y, obs_x, obs_y):
    """Quantitative misfit (task 2.2): median orthogonal distance from each
    observed point to the model polyline in the (x, y) ratio plane, each axis
    normalized by the observed data range so both contribute comparably.

    NB this is a DISPLAY metric between the curve and the observed (ZIR)
    scatter. The Multistart objective is inv.rmse_to_target -- the RMSE to the
    hand-set TARGET CURVE (Lormand stage 2) -- a deliberately different
    definition; do not conflate the two.
    """
    cx = np.asarray(curve_x, float)
    cy = np.asarray(curve_y, float)
    px = np.asarray(obs_x, float)
    py = np.asarray(obs_y, float)
    ok = np.isfinite(px) & np.isfinite(py)
    px, py = px[ok], py[ok]
    if px.size == 0 or cx.size < 2:
        return None
    sx = np.ptp(px) or 1.0                     # normalize by the DATA range
    sy = np.ptp(py) or 1.0
    cx, cy, px, py = cx / sx, cy / sy, px / sx, py / sy
    ax, ay = cx[:-1], cy[:-1]                  # segment starts
    dx, dy = np.diff(cx), np.diff(cy)          # segment vectors
    seg2 = dx * dx + dy * dy
    seg2[seg2 == 0] = 1e-300
    # (n_pts, n_segs) projection parameter, clamped to the segment
    t = ((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / seg2
    t = np.clip(t, 0.0, 1.0)
    ex = ax + t * dx - px[:, None]
    ey = ay + t * dy - py[:, None]
    return float(np.median(np.min(np.hypot(ex, ey), axis=1)))


# Constant-H2O reference isopleths on the viscosity preview — mirrors the P1
# melt-only main figure (ISO_H2O_WT in ot_viscosity_volatile_overlay.py,
# 2026-08-08; value duplicated on purpose: importing that script would pull its
# pandas/env-path baggage into the frozen exe).
ISO_H2O_WT = [3.0, 4.0, 5.0, 6.0]

# Fixed preview frame, mirroring the P1 manuscript figure (user 2026-08-08;
# ot_viscosity_volatile_overlay.PANEL_YLIM / PANEL_XLIM_F / PANEL_XLIM_T must
# stay in step): log10 eta 1.5-5, F 1.0->0.0, T 1000->750 C. Given in DISPLAY
# order, so they REPLACE invert_xaxis() rather than combining with it.
# UNLIKE the manuscript figure, this panel renders arbitrary parameters on an
# arbitrary T-F curve, so a pinned frame can push a curve out of sight —
# _frame_overflow() announces anything outside instead of silently clipping,
# and the pop-out's axis-range controls override the frame for a closer look.
# y lowered 2->1.5 on 2026-08-10 with the canonical replacement (new central
# dips to log eta 1.96 at T_sat; adopted band edge 1.81) — see overlay comment.
PREVIEW_YLIM = (1.5, 5.0)
PREVIEW_XLIM_F = (1.0, 0.0)
PREVIEW_XLIM_T = (1000.0, 750.0)


def _frame_overflow(le, iso, F, Tv):
    """One-line flag naming what the pinned preview frame cuts off ("" when
    everything fits). The manuscript figure is checked once against its own
    fixed dataset; this panel cannot be, so it has to say so on screen."""
    msgs = []
    ys = [np.asarray(le, float)] + [np.asarray(v, float) for v in iso.values()]
    ys = [a[np.isfinite(a)] for a in ys]
    ys = np.concatenate(ys) if any(a.size for a in ys) else np.array([])
    if ys.size:
        lo, hi = min(PREVIEW_YLIM), max(PREVIEW_YLIM)
        if ys.min() < lo - 1e-9:
            msgs.append(f"log eta down to {ys.min():.2f}")
        if ys.max() > hi + 1e-9:
            msgs.append(f"log eta up to {ys.max():.2f}")
    for arr, lim, nm, dp in ((F, PREVIEW_XLIM_F, "F", 2),
                             (Tv, PREVIEW_XLIM_T, "T", 0)):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        lo, hi = min(lim), max(lim)
        if a.min() < lo - 1e-9 or a.max() > hi + 1e-9:
            msgs.append(f"{nm} spans {a.min():.{dp}f}-{a.max():.{dp}f}")
    return ("outside the fixed frame: " + "; ".join(msgs)) if msgs else ""


def viscosity_preview(state, p9, T_of_F=None, const_T=800.0, iso_h2o=None):
    """Coupled log eta vs MELT FRACTION F from the CURRENT volatile trajectory
    (PLACEHOLDER majors). Evaluated per F step of the forward run: dissolved
    H2O(F) and melt F(F) come straight from the volatile model; T(F) via the
    T-F curve when loaded, else the constant temperature. Returns
    (F, le, le_cf, F_sat, T_sat) with T_sat=None in const-T mode.
    iso_h2o (optional): list of H2O wt% values -> a 6th element is appended,
    a dict {wt: logEta array} of constant-H2O isopleths on the SAME T(F) /
    melt-F path with only H2O frozen (return stays a 5-tuple when omitted)."""
    F = state["F"]
    Tv = T_of_F(F) if T_of_F is not None else np.full(len(F), float(const_T))
    le, le_cf = np.empty(len(F)), np.empty(len(F))
    for i in range(len(F)):
        le[i] = log_eta({**ot_majors, "H2O": state["H2O"][i],
                         "F": state["Fwt"][i]}, Tv[i])
        le_cf[i] = log_eta({**ot_majors, "H2O": p9["H2O_init"],
                            "F": state["Fwt"][i]}, Tv[i])
    if state["sat_i"] is not None:
        F_sat = float(F[state["sat_i"]])
        T_sat = float(Tv[state["sat_i"]]) if T_of_F is not None else None
    else:
        F_sat, T_sat = None, None
    if iso_h2o is None:
        return F, le, le_cf, F_sat, T_sat
    iso = {w: np.array([log_eta({**ot_majors, "H2O": w, "F": state["Fwt"][i]},
                                Tv[i]) for i in range(len(F))])
           for w in iso_h2o}
    return F, le, le_cf, F_sat, T_sat, iso


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ApatiteGUI:
    def __init__(self, root):
        self.root = root
        root.title(f"{APP_NAME} — Apatite explorer")
        root.geometry("1280x800")

        self.obs = None                 # observed ratios dict
        self.tf = None                  # (F, T_K, T_of_F) or None
        self.obs_path = None            # source path of the loaded obs csv
        self.tf_path = None             # source path of the loaded T-F csv
        self.const_T = tk.DoubleVar(value=800.0)
        self.t_mode = tk.StringVar(value="const")   # "path" | "const" | "formula"
        # Huber 2010 T-F formula parameters (defaults = the B2 curve family)
        self.fml_Ti = tk.DoubleVar(value=1000.0)
        self.fml_Tf = tk.DoubleVar(value=750.0)
        self.fml_b = tk.DoubleVar(value=2.0)
        self.logx = tk.BooleanVar(value=False)      # right-panel log-x (task 2.3)
        self.misfit = None              # ZIR misfit of the current curve (task 2.2)
        self.overlay_idx = set()        # run indices overlaid on Eye-fit (task 4.2)
        self.ms_overlay = []            # multistart top-N curves (task 4.3)
        self.env_targets = {"low": None, "central": None, "high": None}
        self.env_curves = {}            # envelope preview curves on Eye-fit
        self._env_cur = None            # envelope target currently running
        self._env_pending = []          # remaining envelope targets
        self._env_results = {}          # name -> sorted multistart results
        self.ms_explore = []            # (rmse, x) explore draws of the last run
        self._fig_dir = None            # working dir for project figures/ (M3-2)
        self.state = None               # last compute_state result
        self._after_id = None           # debounce handle
        self._ms_thread = None
        self._ms_queue = queue.Queue()
        self.ms_results = None          # sorted [(rmse, x), ...]
        self.ms_cmap = tk.StringVar(value=MS_CMAPS[0])   # curve-fan colormap
        self.ms_psize = tk.IntVar(value=12)     # obs marker size (ZIR = 1.5x)
        self._ms_norm = None            # log10-metric Normalize of last draw
        self._ms_popouts = []           # open bundle FigureWindows (restyle)
        self.proj = dict(path=None, runs=[])   # in-memory project (runs = dicts)
        self._proj_tmp = None           # temp dir for embedded data of an opened project

        self._build_statusbar()
        self._build_left_panel()
        self._build_tabs()
        self._load_env_defaults()
        self._draw_fml_preview()
        self._schedule_recompute()
        # M0-4: periodic autosave for crash recovery (only once a project
        # file exists; run completion already autosaves separately)
        self.root.after(AUTOSAVE_MS, self._autosave_tick)

    # ---------------- autosave / crash recovery (spec M0-4) ----------------
    def _autosave_path(self, proj_path):
        return proj_path + ".autosave~"

    def _autosave_tick(self):
        if self.proj.get("path"):
            try:
                self._write_project(self._autosave_path(self.proj["path"]))
            except Exception:
                pass                        # never let autosave break the UI
        self.root.after(AUTOSAVE_MS, self._autosave_tick)

    def _maybe_recover_autosave(self, path):
        """Return the file to actually open: the autosave if it is newer and
        the user opts in, else the project itself."""
        ap = self._autosave_path(path)
        try:
            if (os.path.exists(ap)
                    and os.path.getmtime(ap) > os.path.getmtime(path) + 1
                    and messagebox.askyesno(
                        "Crash recovery",
                        "A newer autosave of this project exists (probably "
                        "from an interrupted session).\nOpen the autosave "
                        "instead?")):
                return ap
        except OSError:
            pass
        return path

    # ---------------- status bar (v2 C4) ----------------
    def _build_statusbar(self):
        """Global bottom strip: project name + the forward/status messages
        that used to sit as a small label inside the left panel."""
        sb = ttk.Frame(self.root, padding=(8, 2, 8, 3))
        sb.pack(side="bottom", fill="x")
        self.sb_proj = ttk.Label(sb, text="(no project)", foreground="#555555")
        self.sb_proj.pack(side="left")
        ttk.Separator(sb, orient="vertical").pack(side="left", fill="y",
                                                  padx=8, pady=2)
        self.status = ttk.Label(sb, text="", foreground="#006400")
        self.status.pack(side="left", fill="x")

    # ---------------- left panel ----------------
    def _build_left_panel(self):
        # fixed-width; the controls above scroll (5.4), the readout below is
        # pinned OUTSIDE the scroll canvas so the live numbers are always
        # visible without scrolling (v2 A2 — at 1280x800 they sat below the
        # fold inside the canvas)
        outer = ttk.Frame(self.root)
        outer.pack(side="left", fill="y")
        bottom = ttk.Frame(outer, padding=(6, 0, 6, 4))
        bottom.pack(side="bottom", fill="x")
        cvs = tk.Canvas(outer, highlightthickness=0, width=340)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cvs.pack(side="left", fill="y", expand=True)
        left = ttk.Frame(cvs, padding=6)
        cvs.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>",
                  lambda e: cvs.configure(scrollregion=cvs.bbox("all"),
                                          width=left.winfo_reqwidth()))

        def _left_wheel(e):
            """Scroll the panel only when the pointer is inside it (entries
            consume their own wheel events first via 'break')."""
            w = self.root.winfo_containing(e.x_root, e.y_root)
            while w is not None:
                if w in (left, cvs):
                    cvs.yview_scroll(-1 if e.delta > 0 else 1, "units")
                    return
                w = w.master
        self.root.bind("<MouseWheel>", _left_wheel, add="+")

        # --- project ---
        f_proj = ttk.LabelFrame(left, text="Project", padding=4)
        f_proj.pack(fill="x", pady=(0, 6))
        self.lbl_proj = ttk.Label(f_proj, text="(no project)", width=42)
        self.lbl_proj.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(f_proj, text="Open...", width=8,
                   command=self._open_project_dialog).grid(row=1, column=0)
        ttk.Button(f_proj, text="Save", width=8,
                   command=self._save_project).grid(row=1, column=1)
        ttk.Button(f_proj, text="Save As...", width=9,
                   command=lambda: self._save_project(ask=True)).grid(row=1, column=2)
        ttk.Button(f_proj, text="Snapshot current as run",
                   command=self._snapshot_run).grid(row=2, column=0,
                                                    columnspan=3, pady=(4, 0))

        # --- data files ---
        f_files = ttk.LabelFrame(left, text="Data", padding=4)
        f_files.pack(fill="x", pady=(0, 6))
        self.lbl_obs = ttk.Label(f_files, text="observed: (none)", width=34)
        self.lbl_obs.grid(row=0, column=0, sticky="w")
        ttk.Button(f_files, text="Load...", width=7,
                   command=self._pick_obs).grid(row=0, column=1)
        self.lbl_tf = ttk.Label(f_files, text="T-F curve: (const T)", width=34)
        self.lbl_tf.grid(row=1, column=0, sticky="w")
        ttk.Button(f_files, text="Load...", width=7,
                   command=self._pick_tf).grid(row=1, column=1)
        # temperature mode -- explicit radio choice (task 5.1); selecting
        # "constant T" keeps a loaded T-F file, it just deactivates it
        rowm = ttk.Frame(f_files)
        rowm.grid(row=2, column=0, columnspan=2, sticky="w")
        self.rb_path = ttk.Radiobutton(rowm, text="T-F path (from file)",
                                       variable=self.t_mode, value="path",
                                       command=self._on_tmode)
        self.rb_path.pack(side="left")
        self.rb_const = ttk.Radiobutton(rowm, text="constant T",
                                        variable=self.t_mode, value="const",
                                        command=self._on_tmode)
        self.rb_const.pack(side="left", padx=(8, 0))
        rowm2 = ttk.Frame(f_files)
        rowm2.grid(row=3, column=0, columnspan=2, sticky="w")
        self.rb_formula = ttk.Radiobutton(
            rowm2, text="T-F formula (Huber 2010)",
            variable=self.t_mode, value="formula", command=self._on_tmode)
        self.rb_formula.pack(side="left")
        row = ttk.Frame(f_files)
        row.grid(row=4, column=0, columnspan=2, sticky="w")
        self.lbl_constT = ttk.Label(row, text="const T (degC):")
        self.lbl_constT.pack(side="left")
        self.ent_constT = ttk.Entry(row, textvariable=self.const_T, width=7)
        self.ent_constT.pack(side="left")
        self.ent_constT.bind("<Return>", lambda *_: self._schedule_recompute())
        ttk.Button(row, text="clear TF", width=8,
                   command=self._clear_tf).pack(side="left", padx=4)

        # T-F formula section: generate the cooling path from the Huber 2010
        # law instead of loading a digitized CSV (same downstream contract)
        fml = CollapsibleFrame(f_files, "T-F formula (Huber 2010)")
        fml.grid(row=5, column=0, columnspan=2, sticky="we", pady=(2, 0))
        self.fml_frame = fml
        self.fml_rows = {}              # name -> (label, entry) for greying
        for i, (name, var, unit) in enumerate(
                (("T_initial", self.fml_Ti, " (degC)"),
                 ("T_final", self.fml_Tf, " (degC)"),
                 ("b", self.fml_b, ""))):
            lab = ttk.Label(fml.body, text=name + unit, width=15)
            lab.grid(row=i, column=0, sticky="w")
            info = PARAM_INFO[name]
            Tooltip(lab, f"{name}: {info['desc']}\n{info['range']}"
                         f"\nref: {info['ref']}")
            e = ttk.Entry(fml.body, textvariable=var, width=8)
            e.grid(row=i, column=1, sticky="w")
            e.bind("<Return>", lambda *_: self._on_formula_change())
            e.bind("<FocusOut>", lambda *_: self._on_formula_change())
            self.fml_rows[name] = (lab, e)
        # mini preview of the generated T-F curve (plain Canvas, no mpl)
        self.fml_canvas = tk.Canvas(fml.body, width=180, height=84,
                                    highlightthickness=0, background="white")
        self.fml_canvas.grid(row=0, column=2, rowspan=3, padx=(8, 0))
        b_csv = ttk.Button(fml.body, text="Export CSV", width=11,
                           command=self._export_formula_csv)
        b_csv.grid(row=3, column=0, columnspan=3, sticky="w", pady=(3, 0))
        Tooltip(b_csv, "save the generated curve as a Var1(T_C)/Var2(F) CSV "
                       "(same format the file loader and the inversion "
                       "wrappers read)")
        fml.set_collapsed(True)
        self._update_tmode_widgets()

        # --- parameters (grouped, collapsible -- tasks 1.1/1.2) ---
        f_par = ttk.LabelFrame(left, text="Forward parameters", padding=4)
        f_par.pack(fill="x", pady=(0, 6))
        self.pvars = {}
        self.kspec = tk.DoubleVar(value=DEFAULT_KSPEC)
        self.groups = {}                # title -> CollapsibleFrame

        def param_row(parent, i, name, var, lo, hi):
            """label | [-] | lo | slider | hi | [+] | entry ; +/- nudge by
            range/NUDGE_DIV (per-parameter step), clamped to the slider range."""
            step = (hi - lo) / NUDGE_DIV
            lab = ttk.Label(parent, text=name, width=11)
            lab.grid(row=i, column=0, sticky="w")
            info = PARAM_INFO.get(name)
            if info:
                Tooltip(lab, f"{name}: {info['desc']}\n{info['range']}"
                             f"\nref: {info['ref']}")
            bm = ttk.Button(parent, text="-", width=3,
                            command=lambda: self._nudge(var, -step, lo, hi))
            bm.grid(row=i, column=1)
            bm.bind("<Control-Button-1>",           # Ctrl+click = 10x step
                    lambda e: self._nudge(var, -10 * step, lo, hi) or "break")
            ttk.Label(parent, text=f"{lo:g}", font=small_font(),
                      foreground="#888888").grid(row=i, column=2, sticky="e")
            ttk.Scale(parent, from_=lo, to=hi, variable=var, length=104,
                      command=lambda *_: self._schedule_recompute()
                      ).grid(row=i, column=3)
            ttk.Label(parent, text=f"{hi:g}", font=small_font(),
                      foreground="#888888").grid(row=i, column=4, sticky="w")
            bp = ttk.Button(parent, text="+", width=3,
                            command=lambda: self._nudge(var, +step, lo, hi))
            bp.grid(row=i, column=5)
            bp.bind("<Control-Button-1>",
                    lambda e: self._nudge(var, +10 * step, lo, hi) or "break")
            e = ttk.Entry(parent, textvariable=var, width=8)
            e.grid(row=i, column=6, padx=(4, 0))
            e.bind("<Return>", lambda *_: self._schedule_recompute())
            e.bind("<MouseWheel>",                  # wheel = one step (task 1.3)
                   lambda ev: self._nudge(var, step if ev.delta > 0 else -step,
                                          lo, hi) or "break")

        for title, names in PARAM_GROUPS:
            grp = CollapsibleFrame(f_par, title)
            grp.pack(fill="x")
            self.groups[title] = grp
            for i, k in enumerate(names):
                if k == "K_spec":
                    var, (lo, hi) = self.kspec, KSPEC_RANGE
                else:
                    lo, hi = PARAM_RANGE[k]
                    var = tk.DoubleVar(value=DEFAULT_PARAMS[k])
                    self.pvars[k] = var
                param_row(grp.body, i, k, var, lo, hi)

        btns = ttk.Frame(f_par)
        btns.pack(pady=(6, 0))
        ttk.Button(btns, text="Reset", command=self._reset).pack(side="left")
        ttk.Button(btns, text="Load params CSV",
                   command=self._load_params_csv).pack(side="left", padx=4)
        b_exp = ttk.Button(btns, text="Export bundle",
                           command=self._save_target)
        b_exp.pack(side="left")
        Tooltip(b_exp, "exports/YYYYMMDD_HHMMSS/: target curve CSV + params "
                       "CSV + target-curve figure (PNG 300 dpi + PDF)")

        # --- readout: structured key-value summary (Phase 3) ---
        # lives in `bottom` (pinned under the scroll canvas, always visible)
        ttk.Separator(bottom, orient="horizontal").pack(fill="x", pady=(2, 3))
        self.readout = ttk.Treeview(bottom, columns=("k", "v"), show="",
                                    height=11, selectmode="none")
        self.readout.column("k", width=140, anchor="w", stretch=False)
        self.readout.column("v", width=180, anchor="w")
        self.readout.tag_configure("warn", foreground="#b45309")
        self.readout.pack(fill="x")
        det = CollapsibleFrame(bottom, "details (raw log)")
        det.pack(fill="x")
        self.readout_raw = tk.Text(det.body, width=42, height=8,
                                   font=("Consolas", 8), state="disabled")
        self.readout_raw.pack(fill="x")
        det.set_collapsed(True)

    def _attach_nav(self, canvas, parent):
        """Compact matplotlib zoom/pan toolbar, right-aligned in a tab's bar
        strip (v2 D2) — the popout FigureWindows always had one, the embedded
        tabs did not. The hover x,y readout is suppressed: it constantly
        resized the toolbar."""
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        tb.set_message = lambda s: None
        tb.update()
        tb.pack(side="right", padx=(0, 2))
        return tb

    # ---------------- tabs ----------------
    def _build_tabs(self):
        nb = ttk.Notebook(self.root)
        nb.pack(side="left", fill="both", expand=True)
        self.nb = nb

        def make_fig_tab(title, nrows=1, ncols=2):
            frame = ttk.Frame(nb)
            nb.add(frame, text=title)
            bar = ttk.Frame(frame)             # per-tab controls strip
            bar.pack(fill="x")
            fig = Figure(figsize=(8, 5), dpi=96)
            axes = fig.subplots(nrows, ncols)
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._attach_nav(canvas, bar)      # zoom/pan on every tab (v2 D2)
            return bar, fig, axes, canvas

        # tab renamed "Eye-fit" -> "Explore target curve" (user, 2026-08-04);
        # internal names/methods keep the eyefit stem
        bar1, self.fig1, self.ax1, self.cv1 = make_fig_tab(
            "Explore target curve", 1, 2)
        self.cv1.mpl_connect("pick_event", self._on_pick)
        ttk.Button(bar1, text="Plot", width=7,
                   command=self._eyefit_plot).pack(side="left", padx=(6, 2),
                                                   pady=2)
        ttk.Button(bar1, text="Clear", width=7,
                   command=self._eyefit_clear).pack(side="left", padx=2, pady=2)
        # v2 C3: one toolbar row — Plot/Clear/log X together on the left,
        # ZIR misfit (v2 A3: moved out of the axes) on the right
        ttk.Separator(bar1, orient="vertical").pack(side="left", fill="y",
                                                    padx=6, pady=3)
        ttk.Checkbutton(bar1, text="log X (XF/XCl panel)", variable=self.logx,
                        command=lambda: self._draw_eyefit()
                        if self.state is not None else None
                        ).pack(side="left")
        self.lbl_misfit = ttk.Label(bar1, text="", foreground="#555555")
        self.lbl_misfit.pack(side="right", padx=(0, 10))
        bar2, self.fig2, self.ax2, self.cv2 = make_fig_tab("Melt volatiles",
                                                           1, 1)
        ttk.Button(bar2, text="Pop out", width=8,
                   command=lambda: self._popout_preview("volatiles")
                   ).pack(side="left", padx=(6, 2), pady=2)
        bar3, self.fig3, ax3, self.cv3 = make_fig_tab("Viscosity preview",
                                                      1, 2)
        self.ax3f, self.ax3t = ax3      # (a) vs melt F | (b) vs T, shared y
        self.ax3t.sharey(self.ax3f)     # (two-panel layout, user 2026-08-08)
        ttk.Button(bar3, text="Pop out", width=8,
                   command=lambda: self._popout_preview("viscosity")
                   ).pack(side="left", padx=(6, 2), pady=2)
        self._build_multistart_tab()
        from apatite_m1 import M1Panel          # M1 module (P1 spec §5)
        self.m1 = M1Panel(nb, self)
        nb.add(self.m1, text="Apatite data (M1)")
        from apatite_smelt_panel import SmeltPanel   # melt S (Tassara 2026)
        self.smelt = SmeltPanel(nb, self)
        nb.add(self.smelt, text="Melt S")

    def _build_multistart_tab(self):
        frame = ttk.Frame(self.nb, padding=6)
        self.nb.add(frame, text="Multistart")
        top = ttk.Frame(frame)
        top.pack(fill="x")

        # bounds table — TWO column-pairs (v2 C1: 5+4 params side by side,
        # half the height of the old single column so the figure below gets
        # the vertical space back)
        f_b = ttk.LabelFrame(top, text="BOUNDS (lb / ub)", padding=4)
        f_b.pack(side="left", fill="y")
        self.bvars = {}
        for i, k in enumerate(PARAM_ORDER):
            r, base = (i, 0) if i < 5 else (i - 5, 3)
            ttk.Label(f_b, text=k, width=11).grid(row=r, column=base,
                                                  sticky="w",
                                                  padx=(0 if base == 0 else 10,
                                                        0))
            lb = tk.DoubleVar(); ub = tk.DoubleVar()
            self.bvars[k] = (lb, ub)
            ttk.Entry(f_b, textvariable=lb, width=7).grid(row=r, column=base + 1)
            ttk.Entry(f_b, textvariable=ub, width=7).grid(row=r, column=base + 2)
        ttk.Button(f_b, text="center on current params (+/-30%)",
                   command=self._center_bounds).grid(row=5, column=0,
                                                     columnspan=6,
                                                     pady=(6, 0))
        self._center_bounds()

        # run controls
        f_r = ttk.LabelFrame(top, text="Run", padding=6)
        f_r.pack(side="left", fill="y", padx=8)
        self.n_start = tk.IntVar(value=100)
        self.n_jobs = tk.IntVar(value=max(1, round(0.8 * (os.cpu_count() or 1))))
        self.rmse_mult = tk.DoubleVar(value=5.0)
        self.topn = tk.IntVar(value=5)          # Eye-fit top-N overlay (4.3)
        # tolerance: engine-tight 1e-10 default; 1e-2 = the Lormand MATLAB GUI
        # 'Tolerance' -> runs stop early and fan out over the data (display!).
        self.ms_ftol = tk.StringVar(value="1e-10")
        # explore draws: N random bounds samples forwarded WITHOUT optimisation,
        # drawn under the fitted fan (prior-predictive curve space of the bounds).
        self.explore_n = tk.IntVar(value=0)
        # v2 C1: settings in TWO label/entry column-pairs; the four
        # results-oriented buttons move to a horizontal strip below
        for i, (lab, var, wid) in enumerate(
                (("starts", self.n_start, 7),
                 ("processes", self.n_jobs, 7),
                 ("accept <= x best", self.rmse_mult, 7),
                 ("top-N overlay", self.topn, 7),
                 ("tolerance (ftol)", self.ms_ftol, 7),
                 ("explore draws", self.explore_n, 7))):
            r, base = (i, 0) if i < 3 else (i - 3, 2)
            ttk.Label(f_r, text=lab).grid(row=r, column=base, sticky="w",
                                          padx=(0 if base == 0 else 10, 0))
            ttk.Entry(f_r, textvariable=var, width=wid).grid(row=r,
                                                             column=base + 1)
        # objective metric (P1 spec §7.4): logRMSE default per user 2026-07-04.
        # NB pre-existing runs (incl. the 26.6.25 canonical) used plain "rmse" —
        # the two are NOT comparable; every display/export states the metric.
        ttk.Label(f_r, text="objective").grid(row=3, column=0, sticky="w")
        self.ms_metric = tk.StringVar(value="logrmse")
        ttk.Combobox(f_r, textvariable=self.ms_metric, width=8, state="readonly",
                     values=("logrmse", "rmse")).grid(row=3, column=1)
        ttk.Label(f_r, text="run note").grid(row=3, column=2, sticky="w",
                                             padx=(10, 0))
        self.run_note = tk.StringVar(value="")
        ttk.Entry(f_r, textvariable=self.run_note, width=7).grid(row=3,
                                                                 column=3)
        self.btn_run = ttk.Button(f_r, text="Run multistart",
                                  command=self._run_multistart)
        self.btn_run.grid(row=4, column=0, columnspan=2, pady=(8, 0),
                          sticky="we")
        self.pbar = ttk.Progressbar(f_r, length=140, maximum=100)
        self.pbar.grid(row=4, column=2, columnspan=2, pady=(8, 0),
                       padx=(10, 0), sticky="we")
        self.lbl_ms = ttk.Label(f_r, text="idle")
        self.lbl_ms.grid(row=5, column=0, columnspan=4, pady=(2, 0))

        # v2 C1: results-oriented actions on one strip under the control row
        acts = ttk.Frame(frame)
        acts.pack(fill="x", pady=(6, 0))
        for txt, cmd in (("Export Multioutput CSV", self._export_multioutput),
                         ("Save figures", self._save_ms_figures),
                         ("Apply best fit to sliders", self._apply_best),
                         ("RMSE diagnostics window", self._open_rmse_diag),
                         ("Pop out bundle window", self._popout_bundle)):
            ttk.Button(acts, text=txt, command=cmd).pack(side="left",
                                                         padx=(0, 6))

        # 2026-08-04 layout: the results row and the figure live in a vertical
        # PanedWindow — drag the sash to trade table space for figure space;
        # the figure pane takes all extra height (weight=1), so the three
        # panels are finally big by default
        pw = ttk.PanedWindow(frame, orient="vertical")
        pw.pack(fill="both", expand=True, pady=(6, 0))
        reswrap = ttk.Frame(pw)             # results row + full-width notes
        pw.add(reswrap, weight=0)
        f_res_row = ttk.Frame(reswrap)
        f_res_row.pack(fill="both", expand=True)

        # envelope: 3 bracketing targets (26.6.24 honest-uncertainty workflow)
        f_e = ttk.LabelFrame(f_res_row, text="Envelope (3 targets)", padding=4)
        f_e.pack(side="right", fill="y", padx=(6, 0))
        self.env_lbls = {}
        for r, n in enumerate(("low", "central", "high")):
            ttk.Button(f_e, text=f"set {n}", width=10,
                       command=lambda n=n: self._env_set(n)).grid(
                row=r, column=0, sticky="w")
            lbl = ttk.Label(f_e, text="-", font=("Consolas", 8))
            lbl.grid(row=r, column=1, sticky="w", padx=(4, 0))
            self.env_lbls[n] = lbl
        ttk.Button(f_e, text="suggest low/high from central",
                   command=self._env_suggest).grid(row=3, column=0, columnspan=2,
                                                   pady=(6, 0), sticky="we")
        ttk.Button(f_e, text="preview on target tab",
                   command=self._env_preview).grid(row=4, column=0, columnspan=2,
                                                   sticky="we")
        self.btn_env = ttk.Button(f_e, text="Run envelope (3x multistart)",
                                  command=self._run_envelope)
        self.btn_env.grid(row=5, column=0, columnspan=2, pady=(6, 0), sticky="we")
        ttk.Label(f_e, text="union of the 3 accepted families\n= honest melt-"
                            "volatile range", font=small_font(),
                  foreground="#555555").grid(row=6, column=0, columnspan=2,
                                             pady=(4, 0))

        # project run history
        f_h = ttk.LabelFrame(f_res_row, text="Project runs", padding=4)
        f_h.pack(side="right", fill="both", padx=(6, 0))
        self.runs_list = tk.Listbox(f_h, width=28, height=6,
                                    font=("Consolas", 8))
        self.runs_list.pack(fill="both", expand=True)
        rowb = ttk.Frame(f_h)
        rowb.pack(pady=(4, 0))
        ttk.Button(rowb, text="Load selected",
                   command=self._load_run).pack(side="left")
        ttk.Button(rowb, text="Add to overlay",
                   command=self._add_overlay).pack(side="left", padx=4)
        ttk.Button(rowb, text="Clear overlays",
                   command=self._clear_overlays).pack(side="left")

        # results: headline + param TABLE (v2 A1 — was a tk.Text squeezed into
        # a ~140 px sliver that wrapped into unreadable fragments) + notes.
        # Left column of the second row; envelope + runs sit to its right.
        f_res = ttk.Frame(f_res_row)
        f_res.pack(side="left", fill="both", expand=True)
        self.ms_head = ttk.Label(f_res, text="no results yet",
                                 font=section_font())
        self.ms_head.pack(anchor="w")
        # created unpacked: stays invisible until the first results (A4).
        # 6 visible rows + its own vertical scrollbar keep the pane short;
        # drag the sash (or scroll) for the rest
        self._ms_tblf = tblf = ttk.Frame(f_res)
        self.ms_table = ttk.Treeview(tblf, columns=(), show="headings",
                                     height=6, selectmode="none")
        vs = ttk.Scrollbar(tblf, orient="vertical",
                           command=self.ms_table.yview)
        hs = ttk.Scrollbar(tblf, orient="horizontal",
                           command=self.ms_table.xview)
        self.ms_table.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        self.ms_table.tag_configure("rail", foreground="#b45309")
        self.ms_table.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="we")
        tblf.columnconfigure(0, weight=1)
        # notes: FULL-WIDTH line under the whole results row (in the narrow
        # results column they clipped); "\n" reserves the 2-line height when
        # the pane measures itself at add time
        self.ms_notes = ttk.Label(reswrap, text="\n", justify="left",
                                  foreground="#555555", wraplength=1320)
        self.ms_notes.pack(anchor="w", fill="x", pady=(3, 0))

        # figure pane: canvas + caption BELOW the three panels (2026-08-04 —
        # the in-axes titles above the left/right panels moved down here)
        figf = ttk.Frame(pw)
        pw.add(figf, weight=1)
        self.fig4 = Figure(figsize=(8, 3.8), dpi=96)
        self.ax4 = self.fig4.subplots(1, 3)
        self.ms_caption = ttk.Label(figf, text="", foreground="#555555",
                                    anchor="center")
        self.ms_caption.pack(side="bottom", fill="x")
        cv = FigureCanvasTkAgg(self.fig4, figf)
        cv.get_tk_widget().pack(fill="both", expand=True)
        self.cv4 = cv
        self._attach_nav(cv, acts)             # zoom/pan for the bundle (D2)
        # colormap for the rmse-coloured curve fan (2026-08-04): selector on
        # the actions strip AND in the popout bundle window; recolours the
        # existing artists in place (no forward re-runs)
        cmb = ttk.Combobox(acts, textvariable=self.ms_cmap, width=8,
                           state="readonly", values=MS_CMAPS)
        cmb.pack(side="right", padx=(0, 8))
        ttk.Label(acts, text="colormap").pack(side="right", padx=(10, 2))
        cmb.bind("<<ComboboxSelected>>", self._recolor_ms)
        # observed-point size (2026-08-04) — restyles in place, like colormap
        spn = ttk.Spinbox(acts, textvariable=self.ms_psize, width=4,
                          from_=2, to=60, increment=2,
                          command=self._resize_ms_points)
        spn.pack(side="right")
        ttk.Label(acts, text="pt size").pack(side="right", padx=(10, 2))
        spn.bind("<Return>", self._resize_ms_points)
        # v2 A4: no bare 0..1 axes before the first run — placeholder instead
        for ax in self.ax4:
            ax.set_visible(False)
        self._ms_placeholder = self.fig4.text(
            0.5, 0.5, "Run multistart to see results here",
            ha="center", va="center", fontsize=11, color="#999999")

    def _fill_ms_table(self, cols, widths, rows, tags=None):
        """(Re)configure the results Treeview: cols/widths/rows; per-row tag."""
        if not self._ms_tblf.winfo_manager():       # first results: show table
            self._ms_tblf.pack(fill="x")
        tv = self.ms_table
        tv.delete(*tv.get_children())
        tv.configure(columns=cols)
        for c, w in zip(cols, widths):
            tv.heading(c, text=c)
            anchor = "w" if c in ("param", "target") else "e"
            tv.column(c, width=w, anchor=anchor, stretch=False)
        for i, row in enumerate(rows):
            tag = (tags[i],) if tags and tags[i] else ()
            tv.insert("", "end", values=row, tags=tag)
        tv.configure(height=max(4, min(len(rows), 12)))

    def _ms_axes_on(self):
        """First results display: swap the A4 placeholder for the real axes."""
        if self._ms_placeholder is not None:
            self._ms_placeholder.remove()
            self._ms_placeholder = None
        for ax in self.ax4:
            ax.set_visible(True)

    def _cmap_extras(self, bar):
        """Bundle style controls for a popout bar (FigureWindow extras):
        colormap selector + observed-point size, shared with the tab."""
        cmb = ttk.Combobox(bar, textvariable=self.ms_cmap, width=8,
                           state="readonly", values=MS_CMAPS)
        cmb.pack(side="right", padx=(0, 8))
        ttk.Label(bar, text="colormap").pack(side="right", padx=(10, 2))
        cmb.bind("<<ComboboxSelected>>", self._recolor_ms)
        spn = ttk.Spinbox(bar, textvariable=self.ms_psize, width=4,
                          from_=2, to=60, increment=2,
                          command=self._resize_ms_points)
        spn.pack(side="right")
        ttk.Label(bar, text="pt size").pack(side="right", padx=(10, 2))
        spn.bind("<Return>", self._resize_ms_points)

    def _obs_size(self):
        """Current observed-point size (pt², matrix; ZIR draws 1.5x)."""
        try:
            return max(2, int(self.ms_psize.get()))
        except (tk.TclError, ValueError):
            return 12

    def _ms_surfaces(self):
        """(axes, canvas, colorbar|None) for the embedded bundle and every
        live popout; prunes closed popouts as a side effect."""
        yield self.ax4, self.cv4, getattr(self, "_ms_cbar", None)
        for w in list(self._ms_popouts):
            try:
                if w.winfo_exists():
                    yield w.axes, w.canvas, getattr(w, "_ms_cbar", None)
                else:
                    self._ms_popouts.remove(w)
            except Exception:
                self._ms_popouts.remove(w)

    def _recolor_ms(self, *_):
        """Apply the selected colormap to every metric-coloured curve —
        embedded bundle + open popouts — by restyling the existing artists
        (tagged `_lgval` at draw time); no forward model re-runs."""
        if self._ms_norm is None:
            return
        import matplotlib as mpl
        import matplotlib.cm as mcm
        cmap = mpl.colormaps[self.ms_cmap.get()]
        norm = self._ms_norm
        for axes, canvas, cbar in self._ms_surfaces():
            n = 0
            for ax in axes:
                for ln in ax.get_lines():
                    v = getattr(ln, "_lgval", None)
                    if v is not None:
                        ln.set_color(cmap(norm(v)))
                        n += 1
                # legend handles are snapshots — rebuild so the top-N swatch
                # follows the recolour (popout axes[0] only)
                if n and ax.get_legend() is not None:
                    ax.legend(fontsize=6)
            if cbar is not None:
                cbar.update_normal(mcm.ScalarMappable(norm=norm, cmap=cmap))
            if n:
                canvas.draw_idle()

    def _mark_topn_cbar(self, cbar, lg, n_top):
        """Orange dashed threshold at the N-th best log10(metric) on a metric
        colorbar — everything on the better (lower) side is top-N; same
        visual language as the histogram acceptance cutoff (user 2026-08-06).
        Previous marks are removed first (the embedded colorbar is reused
        across runs); the artists live on cbar.ax, which update_normal
        recolouring does not clear."""
        for art in getattr(cbar, "_topn_marks", []):
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        cbar._topn_marks = []
        if n_top < 1 or n_top > len(lg):
            return
        import matplotlib.patheffects as mpe
        y = float(lg[n_top - 1])
        ln = cbar.ax.axhline(y, color=WONG["orange"], ls="--", lw=1)
        # vertical label INSIDE the bar just above the line (worse side);
        # white halo keeps it readable on any colormap end
        yoff = y + 0.015 * (cbar.norm.vmax - cbar.norm.vmin)
        txt = cbar.ax.text(0.5, yoff, f"top {n_top}", rotation=90,
                           ha="center", va="bottom", fontsize=6,
                           color=WONG["orange"],
                           transform=cbar.ax.get_yaxis_transform(),
                           clip_on=False)
        txt.set_path_effects(
            [mpe.withStroke(linewidth=1.5, foreground="white")])
        cbar._topn_marks = [ln, txt]

    def _resize_ms_points(self, *_):
        """Apply the pt-size spinbox to the observed scatter of the embedded
        bundle + open popouts, in place (tagged `_obs_role` at draw time)."""
        s = self._obs_size()
        for axes, canvas, _cb in self._ms_surfaces():
            n = 0
            for ax in axes:
                for coll in ax.collections:
                    role = getattr(coll, "_obs_role", None)
                    if role is not None:
                        coll.set_sizes([s if role == "matrix" else 1.5 * s])
                        n += 1
            if n:
                canvas.draw_idle()

    # ---------------- data loading ----------------
    def _load_env_defaults(self):
        obs = os.environ.get("OT_OBS")
        if obs and os.path.exists(obs):
            self._set_obs(obs)
        tf = os.environ.get("OT_TF")
        if tf and os.path.exists(tf):
            self._set_tf(tf)

    def _pick_obs(self):
        p = filedialog.askopenfilename(title="Observed apatite ratios CSV",
                                       filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if p:
            self._set_obs(p)

    def _set_obs(self, path):
        try:
            self.obs = load_observed(path)
        except Exception as e:                                    # show, don't die
            messagebox.showerror("Observed ratios", str(e))
            return
        self.obs_path = path
        n_zir = int(self.obs["is_zir"].sum())
        self.lbl_obs.config(text=f"observed: {os.path.basename(path)} "
                                 f"(n={len(self.obs['cloh'])}, ZIR {n_zir})")
        self._schedule_recompute()

    def _pick_tf(self):
        p = filedialog.askopenfilename(title="T-F curve CSV (Var1=T_C, Var2=F)",
                                       filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if p:
            self._set_tf(p)

    def _set_tf(self, path):
        try:
            self.tf = load_tf_curve(path)
        except Exception as e:
            messagebox.showerror("T-F curve", str(e))
            return
        self.tf_path = path
        F, T_K, _ = self.tf
        self.lbl_tf.config(text=f"T-F: {os.path.basename(path)} "
                                f"({T_K[0]-273.15:.0f}->{T_K[-1]-273.15:.0f}C)")
        self.t_mode.set("path")                # loading a curve activates it
        self._update_tmode_widgets()
        self._schedule_recompute()

    def _clear_tf(self):
        self.tf = None
        self.tf_path = None
        self.lbl_tf.config(text="T-F curve: (const T)")
        self.t_mode.set("const")
        self._update_tmode_widgets()
        self._schedule_recompute()

    # ---------------- temperature mode (task 5.1) ----------------
    def _path_active(self):
        """True when temperatures come from the loaded T-F curve."""
        return self.t_mode.get() == "path" and self.tf is not None

    def _t_varies(self):
        """True when T varies along the path (file curve or Huber formula),
        i.e. the run is non-isothermal and T readouts are meaningful."""
        return self._path_active() or self.t_mode.get() == "formula"

    def _formula_tf(self, n=NUM_F_LIVE):
        """(F, T_K, T_of_F) from the current Huber-formula entries; raises
        tk.TclError (entry mid-edit) or ValueError (bad parameters)."""
        return formula_tf_curve(float(self.fml_Ti.get()),
                                float(self.fml_Tf.get()),
                                float(self.fml_b.get()), n)

    def _formula_values(self):
        """Current formula parameters as a plain dict (None while an entry
        is mid-edit / non-numeric) -- for run records and the manifest."""
        try:
            return dict(T_initial=float(self.fml_Ti.get()),
                        T_final=float(self.fml_Tf.get()),
                        b=float(self.fml_b.get()))
        except (tk.TclError, ValueError):
            return None

    def _on_formula_change(self):
        self._draw_fml_preview()
        if self.t_mode.get() == "formula":
            self._schedule_recompute()

    def _draw_fml_preview(self):
        """Mini T-F preview of the generated curve (plain tk.Canvas)."""
        cv = self.fml_canvas
        cv.delete("all")
        w, h = int(cv["width"]), int(cv["height"])
        try:
            F, T_K, _ = self._formula_tf(60)
        except Exception:
            cv.create_text(w // 2, h // 2, text="invalid parameters",
                           fill="#a00000", font=("TkDefaultFont", 8))
            return
        ml, mr, mt, mb = 34, 6, 8, 14
        T_C = T_K - 273.15
        x = ml + (F[0] - F) / (F[0] - F[-1]) * (w - ml - mr)  # F 1 -> 0.1
        tmin, tmax = float(T_C.min()), float(T_C.max())
        y = mt + (tmax - T_C) / max(tmax - tmin, 1e-9) * (h - mt - mb)
        cv.create_rectangle(ml, mt, w - mr, h - mb, outline="#cccccc")
        cv.create_line(*np.column_stack([x, y]).ravel().tolist(),
                       fill="#0072B2", width=2)
        small = ("TkDefaultFont", 7)
        cv.create_text(ml - 2, mt + 3, text=f"{tmax:.0f}", anchor="e",
                       font=small, fill="#666666")
        cv.create_text(ml - 2, h - mb - 3, text=f"{tmin:.0f}", anchor="e",
                       font=small, fill="#666666")
        cv.create_text(ml, h - 1, text=f"F={F[0]:g}", anchor="sw",
                       font=small, fill="#666666")
        cv.create_text(w - mr, h - 1, text=f"F={F[-1]:g}", anchor="se",
                       font=small, fill="#666666")

    def _export_formula_csv(self):
        """Save the generated curve as a Var1(T_C)/Var2(F) CSV -- the exact
        format load_tf_curve / build_F_and_T / the inversion wrappers read."""
        try:
            F, T_K, _ = self._formula_tf(NUM_F_LIVE)
        except Exception as e:
            messagebox.showerror("T-F formula", str(e))
            return
        v = self._formula_values()
        init = f"TF_huber_{v['T_initial']:g}-{v['T_final']:g}-b{v['b']:g}.csv"
        p = filedialog.asksaveasfilename(
            title="Export generated T-F curve", defaultextension=".csv",
            initialfile=init,
            initialdir=os.environ.get("OT_OUT", os.getcwd()),
            filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if not p:
            return
        try:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["Var1", "Var2"])
                for t, f in zip(T_K - 273.15, F):
                    w.writerow([f"{t:.6g}", f"{f:.6g}"])
        except Exception as e:
            messagebox.showerror("Export T-F curve", str(e))
            return
        self.status.config(text=f"T-F curve exported: {os.path.basename(p)}")

    def _on_tmode(self):
        """Radio switch. Selecting const T keeps a loaded T-F file (it is only
        deactivated); selecting path without a file (or formula with invalid
        entries) falls back to const T."""
        if self.t_mode.get() == "path" and self.tf is None:
            self.t_mode.set("const")
            self.status.config(text="no T-F curve loaded -- staying in const-T "
                                    "mode (Load... a Var1/Var2 CSV first)",
                               foreground="#a05a00")
        elif self.t_mode.get() == "formula":
            try:
                self._formula_tf(3)
                self.fml_frame.set_collapsed(False)   # show the parameters
            except Exception as e:
                self.t_mode.set("const")
                self.status.config(text="T-F formula invalid -- staying in "
                                        f"const-T mode ({e})",
                                   foreground="#a05a00")
        self._update_tmode_widgets()
        self._schedule_recompute()

    def _update_tmode_widgets(self):
        """Grey out whichever temperature inputs are inactive."""
        mode = self.t_mode.get()
        self.ent_constT.config(state="normal" if mode == "const"
                               else "disabled")
        self.lbl_constT.config(foreground="" if mode == "const" else "#999999")
        self.lbl_tf.config(foreground="" if mode == "path" else "#999999")
        for lab, _e in self.fml_rows.values():
            lab.config(foreground="" if mode == "formula" else "#999999")

    # ---------------- params ----------------
    def _nudge(self, var, delta, lo, hi):
        try:
            v = float(var.get())
        except tk.TclError:                       # entry mid-edit / invalid
            return
        var.set(round(min(hi, max(lo, v + delta)), 6))
        self._schedule_recompute()

    def current_params(self):
        return {k: float(self.pvars[k].get()) for k in PARAM_ORDER}

    def _reset(self):
        for k in PARAM_ORDER:
            self.pvars[k].set(DEFAULT_PARAMS[k])
        self.kspec.set(DEFAULT_KSPEC)
        self._schedule_recompute()

    def _load_params_csv(self):
        p = filedialog.askopenfilename(
            title="Params CSV (Multioutput or bestfit_params)",
            filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if not p:
            return
        try:
            with open(p, newline="", encoding="utf-8-sig") as fh:
                r = list(csv.DictReader(fh))[0]
            if "H2O" in r:                                  # MATLAB Multioutput headers
                vals = {am.PARAM_MAP[k]: float(r[k]) for k in am.PARAM_MAP}
            else:                                           # engine-key headers
                vals = {k: float(r[k]) for k in PARAM_ORDER}
            for k in PARAM_ORDER:
                self.pvars[k].set(vals[k])
            if "K_spec" in r:
                self.kspec.set(float(r["K_spec"]))
        except Exception as e:
            messagebox.showerror("Load params", str(e))
            return
        self._schedule_recompute()

    def _save_target(self):
        """Export bundle (task 4.1): params CSV + target curve CSV + the
        Eye-fit figure as 300-dpi PNG and PDF, into a timestamped folder
        exports/YYYYMMDD_HHMMSS/ under visible_out_base() (OT_OUT > cwd >
        ~\\<app>-output, never the OS temp dir)."""
        if self.state is None:
            return
        base = visible_out_base()
        out = os.path.join(base, "exports",
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            os.makedirs(out, exist_ok=True)
            s = self.state
            data = {"Frac_Melt": s["F"], "XCl_div_XOH": s["cloh"],
                    "XF_div_XOH": s["foh"], "XF_div_XCl": s["fcl"]}
            inv.write_target(data, os.path.join(out, "target_curve.csv"))
            with open(os.path.join(out, "params.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(PARAM_ORDER + ["K_spec"])
                w.writerow([f"{self.current_params()[k]:.6g}"
                            for k in PARAM_ORDER]
                           + [f"{self.kspec.get():.6g}"])
            self.fig1.savefig(os.path.join(out, "eyefit.png"), dpi=300)
            self.fig1.savefig(os.path.join(out, "eyefit.pdf"))
        except Exception as e:
            messagebox.showerror("Export bundle", str(e))
            return
        self.status.config(text=f"exported curve + params + PNG/PDF -> {out}")

    # ---------------- live recompute (debounced) ----------------
    def _schedule_recompute(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(120, self._recompute)

    def _grid(self, n):
        # explicit temperature mode (task 5.1): the T-F curve is used only in
        # "path" mode; in "const" mode a loaded curve stays in memory, inactive
        if self._path_active():
            F, T_K, _ = self.tf
            if len(F) != n:
                F2 = np.linspace(F.max(), max(0.1, F.min()), n)
                order = np.argsort(F)
                return F2, np.interp(F2, F[order], T_K[order])
            return F, T_K
        if self.t_mode.get() == "formula":
            # analytic Huber curve, generated directly at n points; invalid
            # entries raise and surface as "forward failed: ..." in the status
            F, T_K, _ = self._formula_tf(n)
            return F, T_K
        F = np.linspace(1.0, 0.1, n)
        return F, np.full(n, float(self.const_T.get()) + 273.15)

    def _eyefit_clear(self):
        """Eye-fit 'Clear': drop every overlay (run overlays, multistart top-N,
        envelope previews) and blank the two panels until Plot is pressed."""
        self.overlay_idx.clear()
        self.ms_overlay = []
        self.env_curves = {}
        self._refresh_runs_list()
        for ax in np.ravel(self.ax1):
            ax.clear()
            ax.grid(alpha=0.3)
        self.cv1.draw_idle()
        self.status.config(text="target plot cleared — press Plot to redraw",
                           foreground="#006400")

    def _eyefit_plot(self):
        """Eye-fit 'Plot': recompute the forward model from the current sliders
        and redraw all panels (fresh plot, no accumulated overlays removed)."""
        self._recompute()

    def _recompute(self):
        if getattr(self, "_in_recompute", False):   # guard vs re-entrant
            return                                  # slider<->entry callbacks
        self._in_recompute = True
        try:
            self._recompute_inner()
        finally:
            self._in_recompute = False

    def _recompute_inner(self):
        self._after_id = None
        t0 = time.time()
        try:
            F, T_K = self._grid(NUM_F_LIVE)
            self.state = compute_state(self.current_params(),
                                       float(self.kspec.get()), F, T_K)
        except Exception as e:
            self.status.config(text=f"forward failed: {e}", foreground="#a00000")
            return
        # quantitative misfit vs the observed ZIR points (task 2.2)
        self.misfit = None
        if self.obs is not None and self.obs["is_zir"].any():
            mz = self.obs["is_zir"]
            self.misfit = zir_misfit(self.state["foh"], self.state["cloh"],
                                     self.obs["foh"][mz], self.obs["cloh"][mz])
        self._draw_eyefit()
        self._draw_volatiles()
        self._draw_viscosity()
        self._update_readout()
        conv = "" if self.state["conv"] else "  [FIXED-POINT NOT CONVERGED]"
        self.status.config(
            text=f"forward {1e3*(time.time()-t0):.0f} ms{conv}",
            foreground="#a00000" if not self.state["conv"] else "#006400")

    # ---------------- drawing ----------------
    def _plot_obs(self, ax, xkey, on_top=False, size=None):
        """Observed points: y = XCl/XOH, x = XF/XOH or XF/XCl (xkey).

        on_top: draw ABOVE every curve (multistart bundle convention,
        2026-08-04 — the fan used to bury the data). size: matrix marker
        size in pt² (ZIR = 1.5x); default 12/18 (the target-tab look)."""
        if self.obs is None:
            return
        o = self.obs
        x = o["foh"] if xkey == "foh" else (
            o["fcl"] if o["fcl"] is not None else o["foh"] / o["cloh"])
        mz = o["is_zir"]
        s_m = 12 if size is None else size
        z_m, z_z = (6, 6.5) if on_top else (1, 2)
        a_m = ax.scatter(x[~mz], o["cloh"][~mz], s=s_m, c="#bbbbbb",
                         edgecolors="#888888", linewidths=0.4, label="matrix",
                         zorder=z_m, picker=4)
        a_m._obs_idx = np.where(~mz)[0]         # click -> source row (task 2.4)
        a_m._obs_role = "matrix"                # pt-size selector target
        a_z = ax.scatter(x[mz], o["cloh"][mz], s=1.5 * s_m, c=WONG["purple"],
                         marker="D", edgecolors="#5a3350", linewidths=0.4,
                         label="ZIR", zorder=z_z, picker=4)
        a_z._obs_idx = np.where(mz)[0]
        a_z._obs_role = "zir"

    def _overlay_curves(self):
        """[(run_idx, cloh, foh), ...] for the overlaid runs (task 4.2):
        forward-recomputed from each run's stored params/k_spec on the CURRENT
        grid, cached on the run record per grid signature."""
        if not self.overlay_idx:
            return []
        F, T_K = self._grid(NUM_F_LIVE)
        sig = (len(F), float(F[0]), float(F[-1]),
               float(T_K[0]), float(T_K[-1]))
        out = []
        for i in sorted(self.overlay_idx):
            if i >= len(self.proj["runs"]):
                continue
            r = self.proj["runs"][i]
            cache = r.get("_ov")                # transient, never persisted
            if cache is None or cache[0] != sig:
                try:
                    st = compute_state(dict(r["params"]), float(r["k_spec"]),
                                       F, T_K)
                except Exception:
                    continue
                cache = (sig, st["cloh"], st["foh"])
                r["_ov"] = cache
            out.append((i, cache[1], cache[2]))
        return out

    def _on_pick(self, event):
        """Click on an observed point -> small popup with its source CSV row
        (task 2.4)."""
        idx = getattr(event.artist, "_obs_idx", None)
        if idx is None or self.obs is None or self.obs.get("raw") is None \
                or not len(event.ind):
            return
        row = self.obs["raw"].iloc[int(idx[event.ind[0]])]
        items = [(str(k), str(v)) for k, v in row.items()
                 if str(v).strip() not in ("", "nan")][:12]
        txt = "\n".join(f"{k}: {v}" for k, v in items) or "(empty row)"
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{self.root.winfo_pointerx()+12}"
                        f"+{self.root.winfo_pointery()+8}")
        lbl = tk.Label(tip, text=txt, justify="left", relief="solid",
                       borderwidth=1, background="#ffffe0",
                       font=("Consolas", 8), padx=5, pady=3)
        lbl.pack()
        lbl.bind("<Button-1>", lambda *_: tip.destroy())
        tip.after(8000, tip.destroy)            # auto-dismiss

    def _draw_eyefit(self):
        s = self.state
        overlays = self._overlay_curves()
        for ax, xkey, xlab in ((self.ax1[0], "foh", "XF/XOH"),
                               (self.ax1[1], "fcl", "XF/XCl")):
            ax.clear()
            self._plot_obs(ax, xkey)
            for j, (ri, o_cloh, o_foh) in enumerate(overlays):
                ox = o_foh if xkey == "foh" else o_foh / o_cloh
                ax.plot(ox, o_cloh, "--", lw=1.4, alpha=0.6,
                        color=OVERLAY_COLORS[j % len(OVERLAY_COLORS)],
                        label=f"run #{ri+1}", zorder=2.5)
            for m, (m_cloh, m_foh) in enumerate(self.ms_overlay):
                mx = m_foh if xkey == "foh" else m_foh / m_cloh
                ax.plot(mx, m_cloh, "-", color=WONG["orange"], alpha=0.25,
                        lw=1.0, zorder=2.2,
                        label=(f"multistart top-{len(self.ms_overlay)}"
                               if m == 0 else None))
            for n, (e_cloh, e_foh) in self.env_curves.items():
                ex = e_foh if xkey == "foh" else e_foh / e_cloh
                ax.plot(ex, e_cloh, "--", color=ENV_COL[n], lw=1.6,
                        zorder=2.8, label=f"env {n}")
            x = s[xkey]
            ax.plot(x, s["cloh"], "-", color=WONG["blue"], lw=2.0,
                    label="target curve", zorder=3)
            if s["sat_i"] is not None:
                ax.plot(x[s["sat_i"]], s["cloh"][s["sat_i"]], "*", ms=13,
                        color=WONG["orange"], mec="k", mew=0.5,
                        label="H2O-sat onset", zorder=4)
                # onset annotation next to the star (task 2.5)
                lab = f"H2O-sat: F={s['F'][s['sat_i']]:.2f}"
                if self._t_varies():
                    lab += f", T={s['T_K'][s['sat_i']]-273.15:.0f}C"
                ax.annotate(lab, (x[s["sat_i"]], s["cloh"][s["sat_i"]]),
                            textcoords="offset points", xytext=(7, 7),
                            fontsize=8, zorder=4,
                            bbox=dict(fc="white", ec="none", alpha=0.7))
            ax.set_xlabel(xlab)
            ax.set_ylabel("XCl/XOH")
            ax.grid(alpha=0.3)
        # log-x option for the compressed XF/XCl panel (task 2.3)
        self.ax1[1].set_xscale("log" if self.logx.get() else "linear")
        # misfit readout (task 2.2) — on the toolbar row (v2 A3)
        self.lbl_misfit.config(
            text=(f"misfit (ZIR, med. dist.): {self.misfit:.3f}"
                  if self.misfit is not None else ""))
        # single shared figure-level legend instead of two identical
        # per-axes ones (task 2.1)
        if getattr(self, "_fig1_legend", None) is not None:
            self._fig1_legend.remove()
        h, l = self.ax1[0].get_legend_handles_labels()
        self._fig1_legend = self.fig1.legend(h, l, loc="upper center",
                                             ncol=4, fontsize=9)
        self.fig1.tight_layout(rect=(0, 0, 1, 0.93))
        self.cv1.draw_idle()

    def _draw_volatiles(self):
        self._render_volatiles(self.ax2)
        self.fig2.tight_layout()
        self.cv2.draw_idle()

    def _render_volatiles(self, ax):
        """Melt-volatiles panel body — shared by the embedded tab and the
        pop-out window (2026-08-07), which both draw the current state."""
        s = self.state
        ax.clear()
        ax.plot(s["F"], s["H2O"], "-", color=WONG["blue"], lw=2, label="melt H2O")
        ax.plot(s["F"], s["Cl"] * 10, "--", color=WONG["orange"], lw=2,
                label="melt Cl x10")
        ax.plot(s["F"], s["Fwt"] * 10, "-.", color=WONG["green"], lw=2,
                label="melt F x10")
        if s["sat_i"] is not None:
            ax.axvline(s["F"][s["sat_i"]], color=WONG["grey"], ls=":", lw=1.2)
            # x in data coords, y as axes fraction: immune to later ylim
            # changes that used to clip this label at the top spine (v2 A3)
            ax.text(s["F"][s["sat_i"]], 0.97,
                    f" sat F={s['F'][s['sat_i']]:.2f}", fontsize=8,
                    va="top", color=WONG["grey"],
                    transform=ax.get_xaxis_transform())
        ax.set_xlabel("melt fraction F  (crystallisation ->)")
        ax.set_ylabel("wt%  (Cl, F x10)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Melt volatile evolution (current params)")

    def _draw_viscosity(self):
        self._render_viscosity(self.ax3f, self.ax3t)
        self.fig3.tight_layout(rect=(0, 0, 1, 0.93))
        self.cv3.draw_idle()

    def _render_viscosity(self, ax_f, ax_t):
        """Viscosity-preview panels — (a) vs melt fraction F, (b) vs
        temperature — shared by the embedded tab and the pop-out window
        (two-panel layout, user decision 2026-08-08, mirroring the P1
        melt-only main figure). The constant-H2O comparison line is DROPPED
        (same decision); the 3-6 wt% isopleths carry the reference. In
        const-T mode the T panel shows a hint instead (eta(T) degenerates
        to a point there). Axes are PINNED to the manuscript frame
        (PREVIEW_YLIM / PREVIEW_XLIM_F / PREVIEW_XLIM_T, user 2026-08-08);
        anything falling outside is named in an orange in-panel line."""
        ax_f.clear(); ax_t.clear()
        if self._path_active():
            T_of_F = self.tf[2]
        elif self.t_mode.get() == "formula":
            try:
                T_of_F = self._formula_tf(3)[2]
            except Exception:
                T_of_F = None
        else:
            T_of_F = None
        try:
            F, le, _le_cf, F_sat, T_sat, iso = viscosity_preview(
                self.state, self.current_params(), T_of_F,
                float(self.const_T.get()), iso_h2o=ISO_H2O_WT)
        except Exception as e:
            ax_f.text(0.5, 0.5, f"viscosity failed: {e}", ha="center",
                      va="center", fontsize=9, color="#a00000")
            return

        def panel(ax, x, xlabel, satx, satlab, end_labels, xlim):
            ax.set_ylim(*PREVIEW_YLIM)
            ax.plot(x, le, "-", color="#1f3b73", lw=2.2,
                    label="coupled (enriching H2O)")
            # constant-H2O isopleths, same styling as the P1 melt-only figure;
            # wt% end labels only on the OUTER panel (they'd poke into the
            # neighbouring panel otherwise)
            for k, w in enumerate(ISO_H2O_WT):
                ax.plot(x, iso[w], ls=(0, (4, 2.5)), color="0.45", lw=0.9,
                        zorder=1.6,
                        label=(f"constant-H2O ref ({ISO_H2O_WT[0]:g}-"
                               f"{ISO_H2O_WT[-1]:g} wt%)" if k == 0
                               else "_nolegend_"))
                if end_labels:
                    ax.annotate(f"{w:g}", (x[-1], iso[w][-1]), xytext=(3, 0),
                                textcoords="offset points", ha="left",
                                va="center", fontsize=7, color="0.35",
                                annotation_clip=False)
            if satx is not None:
                ax.axvline(satx, color=WONG["grey"], ls="-.", lw=1)
                ax.text(satx, 0.97, satlab, fontsize=8, va="top",
                        color=WONG["grey"],
                        transform=ax.get_xaxis_transform())
            ax.set_xlabel(xlabel)
            ax.set_xlim(*xlim)     # display order -> replaces invert_xaxis()
            ax.grid(alpha=0.3)

        Tv = np.asarray(T_of_F(F), dtype=float) if T_of_F is not None else None
        panel(ax_f, F, "melt fraction F  (crystallisation ->)", F_sat,
              f" F_sat={F_sat:.2f}" if F_sat is not None else "",
              end_labels=(Tv is None), xlim=PREVIEW_XLIM_F)
        ax_f.set_ylabel("log10 eta (Pa s)")
        ax_f.legend(fontsize=7)
        if Tv is not None:
            panel(ax_t, Tv, "temperature (degC)", T_sat,
                  f" T_sat={T_sat:.0f}C" if T_sat is not None else "",
                  end_labels=True, xlim=PREVIEW_XLIM_T)
        else:
            ax_t.text(0.5, 0.5, "constant-T mode:\nload a T-F curve or pick "
                      "the\nHuber formula for the T panel", ha="center",
                      va="center", fontsize=8, color="0.4")
            ax_t.set_xlabel("temperature (degC)")
        ax_t.tick_params(labelleft=False)      # shared y: labels on (a) only
        oof = _frame_overflow(le, iso, F, Tv)
        if oof:                                # never clip in silence
            ax_f.text(0.5, 0.015, oof, transform=ax_f.transAxes, ha="center",
                      va="bottom", fontsize=7, color=WONG["orange"])
        for ax, letter in ((ax_f, "(a)"), (ax_t, "(b)")):
            ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left",
                    va="top", fontsize=9, fontweight="bold")
        mode = (("T from T-F curve" if self._path_active()
                 else "T from Huber formula") if T_of_F is not None
                else f"const T={self.const_T.get():.0f}C")
        ax_f.figure.suptitle(
            f"GRD08 melt viscosity preview ({mode}) -- PLACEHOLDER majors; "
            "H2O>8 wt% extrapolated", fontsize=8)

    def _popout_preview(self, which):
        """Melt-volatiles / viscosity preview in an independent FigureWindow
        (user 2026-08-07): same rendering as the tab, plus the standard
        axis-range controls and a line-style editor (width / colour) for
        figure polish before Save. A snapshot, like the bundle popout — it
        does not follow later parameter edits; reopen for fresh params."""
        if self.state is None:
            messagebox.showinfo("Preview", "no computed state yet")
            return None
        from ot_figwin import FigureWindow
        if which == "volatiles":
            w = FigureWindow(self.root, "Melt volatiles", 1, 1,
                             figsize=(7.5, 5.2), figtype="volatiles",
                             autosave_dir=self.figures_dir(),
                             extras=self._line_style_extras)
            self._render_volatiles(w.axes[0])
            w.fig.tight_layout()
        else:
            # viscosity: 1x2 panels, same layout as the embedded tab
            # (user decision 2026-08-08)
            w = FigureWindow(self.root, "Viscosity preview", 1, 2,
                             figsize=(11.0, 4.8), figtype="viscosity",
                             autosave_dir=self.figures_dir(),
                             extras=self._line_style_extras)
            w.axes[1].sharey(w.axes[0])
            self._render_viscosity(w.axes[0], w.axes[1])
            w.fig.tight_layout(rect=(0, 0, 1, 0.93))
        w.canvas.draw_idle()
        self._note_autosave(w.autosave())
        return w

    def _line_style_extras(self, bar):
        """Line-style editor for a preview popout bar (FigureWindow extras):
        pick a labelled line (or all), set the width, choose a colour —
        restyles the popout's own artists in place and rebuilds the axes
        legend (legend handles are snapshots). The line list fills lazily
        via postcommand: extras build before the first render."""
        w = bar.master                  # the FigureWindow (bar packs inside)

        def lines():
            out = []
            for ax in w.axes:
                for ln in ax.get_lines():
                    lab = ln.get_label()
                    if lab and not lab.startswith("_"):
                        out.append((lab, ln))
            return out

        sel = tk.StringVar(value="all")
        wid = tk.StringVar(value="2")

        def targets():
            pick = sel.get()
            return [ln for lab, ln in lines() if pick in ("all", lab)]

        def refresh():
            for ax in w.axes:
                if ax.get_legend() is not None:
                    ax.legend()
            w.canvas.draw_idle()

        def apply_width(*_):
            try:
                v = float(wid.get())
            except (tk.TclError, ValueError):
                return
            if v <= 0:
                return
            for ln in targets():
                ln.set_linewidth(v)
            refresh()

        def apply_colour(c):
            for ln in targets():
                ln.set_color(c)
            refresh()

        def pick_colour():
            tg = targets()
            if not tg:
                return
            from tkinter import colorchooser
            import matplotlib.colors as mcolors
            c = colorchooser.askcolor(
                parent=w, initialcolor=mcolors.to_hex(tg[0].get_color()))[1]
            if c:
                apply_colour(c)

        def on_select(*_):
            tg = targets()
            if len(tg) == 1:            # show the picked line's width
                wid.set(f"{tg[0].get_linewidth():g}")

        cmb = ttk.Combobox(bar, textvariable=sel, width=16, state="readonly")
        cmb.configure(postcommand=lambda: cmb.configure(
            # dedupe: the two viscosity panels carry same-named lines, and
            # targets() styles every match at once (panels stay in step)
            values=["all"] + list(dict.fromkeys(lab for lab, _ in lines()))))
        cmb.bind("<<ComboboxSelected>>", on_select)
        spn = ttk.Spinbox(bar, textvariable=wid, width=5, from_=0.25, to=8,
                          increment=0.25, command=apply_width)
        spn.bind("<Return>", apply_width)
        btn = ttk.Button(bar, text="colour...", width=8, command=pick_colour)
        # side="right" packs stack right-to-left; this order reads, left to
        # right:  line [combobox]  width [spinbox]  colour...  Save...
        btn.pack(side="right", padx=(0, 8))
        spn.pack(side="right", padx=(2, 6))
        ttk.Label(bar, text="width").pack(side="right")
        cmb.pack(side="right", padx=(2, 6))
        ttk.Label(bar, text="line").pack(side="right", padx=(10, 2))
        # scripted-drive hook: programmatic access to the closures without
        # going through the widgets (colour picker is modal, undriveable)
        w._style = dict(sel=sel, wid=wid, lines=lines, targets=targets,
                        apply_width=apply_width, apply_colour=apply_colour)

    def _update_readout(self):
        """Structured key-value summary (Phase 3) with an automatic
        range-coverage warning; the raw text block lives in the details
        expander below."""
        s = self.state
        rows = []                               # (key, value, tag)
        if self._t_varies():
            src = "T path" if self._path_active() else "T formula"
            rows.append((src,
                         f"{s['T_K'][0]-273.15:.0f}->{s['T_K'][-1]-273.15:.0f}C"
                         f"  (F {s['F'][0]:.2f}->{s['F'][-1]:.2f})", ""))
        else:
            rows.append(("const T", f"{self.const_T.get():.0f}C", ""))
        rows.append(("melt start H2O/Cl/F",
                     f"{s['H2O'][0]:.2f}/{s['Cl'][0]:.3f}/{s['Fwt'][0]:.3f} wt%",
                     ""))
        rows.append(("melt end H2O/Cl/F",
                     f"{s['H2O'][-1]:.2f}/{s['Cl'][-1]:.3f}/{s['Fwt'][-1]:.3f} wt%",
                     ""))
        if s["sat_i"] is not None:
            rows.append(("H2O-sat onset",
                         f"F={s['F'][s['sat_i']]:.2f}"
                         + (f", T={s['T_K'][s['sat_i']]-273.15:.0f}C"
                            if self._t_varies() else ""), ""))
            rows.append(("Cl peak", f"{s['Cl'].max():.3f} wt%", ""))
        else:
            rows.append(("H2O-sat onset", "not reached on the path", "warn"))

        def rng(a):
            return f"{a.min():.3g} .. {a.max():.3g}"

        # model vs observed ranges, plain rows — the "⚠ not covered"
        # coverage warning was removed on user request (2026-08-04): on the
        # real OT data the model range never brackets the full scatter by
        # design, so the marker was permanent noise
        for name, mk in (("Cl/OH", "cloh"), ("F/OH", "foh")):
            m = s[mk]
            rows.append((f"model {name}", rng(m), ""))
            if self.obs is not None and len(self.obs[mk]):
                rows.append((f"obs {name}", rng(self.obs[mk]), ""))
        if self.obs is not None:
            rows.append(("obs n (ZIR)", f"{len(self.obs['cloh'])} "
                         f"({int(self.obs['is_zir'].sum())})", ""))
        rows.append(("misfit (ZIR)",
                     f"{self.misfit:.4g}" if self.misfit is not None else "n/a",
                     ""))
        self.readout.delete(*self.readout.get_children())
        for k, v, tag in rows:
            self.readout.insert("", "end", values=(k, v),
                                tags=(tag,) if tag else ())
        # raw log kept for debugging (details expander)
        raw = ["-- current forward --"] + [f"{k}: {v}" for k, v, _ in rows]
        self.readout_raw.config(state="normal")
        self.readout_raw.delete("1.0", "end")
        self.readout_raw.insert("1.0", "\n".join(raw))
        self.readout_raw.config(state="disabled")

    # ---------------- figures dir + M2 windows (P1 spec) ----------------
    def figures_dir(self):
        """Working dir collected into the project zip under figures/ (M3-2)."""
        if self._fig_dir is None:
            self._fig_dir = tempfile.mkdtemp(prefix="otfigs_")
        return self._fig_dir

    def _popout_bundle(self):
        """Multistart bundle in an independent FigureWindow (spec M2-1/M2-3):
        target curve + all runs coloured by log10(metric) + top-N + observed."""
        if not self.ms_results:
            messagebox.showinfo("Multistart", "run the multistart first")
            return
        from ot_figwin import FigureWindow
        import matplotlib as mpl
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        w = FigureWindow(self.root, "Multistart bundle", 1, 2,
                         figsize=(10.5, 5), figtype="ms_bundle",
                         autosave_dir=self.figures_dir(),
                         extras=self._cmap_extras)
        self._ms_popouts.append(w)      # live recolour on cmap change
        rmses = np.array([r[0] for r in self.ms_results])
        X = np.array([r[1] for r in self.ms_results])
        idx = np.arange(len(rmses))
        if len(idx) > 300:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, 300).astype(int))]
        lg = np.log10(np.maximum(rmses, 1e-12))
        # share the embedded-bundle norm when it exists: colours then match
        # the tab exactly, and _recolor_ms (which always repaints with
        # _ms_norm) cannot shift them on the first colormap change
        norm = (self._ms_norm if self._ms_norm is not None else
                mcolors.Normalize(vmin=lg.min(), vmax=lg.max()))
        cmap = mpl.colormaps[self.ms_cmap.get()]
        F_fit, T_fit = self._grid(NUM_F_FIT)
        try:
            n_top = max(0, int(self.topn.get()))
        except tk.TclError:
            n_top = 5
        for j in idx[::-1]:
            p = inv.coeff_to_params(X[j], 0.0)
            cloh, foh = inv.forward_ratios(p, F_fit, T_fit,
                                           float(self.kspec.get()))
            for a, xx in ((w.axes[0], foh), (w.axes[1], foh / cloh)):
                ln, = a.plot(xx, cloh, "-", color=cmap(norm(lg[j])),
                             alpha=0.5, lw=0.7)
                ln._lgval = lg[j]       # recolourable via the cmap selector
        n_top_eff = min(n_top, len(rmses))
        for j in range(n_top_eff):      # top-N: bold + on top, but colour
            # FOLLOWS the metric colormap (user 2026-08-06 — no separate
            # colour); the colorbar threshold line is what marks them out
            p = inv.coeff_to_params(X[j], 0.0)
            cloh, foh = inv.forward_ratios(p, F_fit, T_fit,
                                           float(self.kspec.get()))
            ln, = w.axes[0].plot(foh, cloh, "-", color=cmap(norm(lg[j])),
                                 lw=1.2, zorder=4,
                                 label="top-N" if j == 0 else None)
            ln._lgval = lg[j]           # recolours with the fan
            ln, = w.axes[1].plot(foh / cloh, cloh, "-",
                                 color=cmap(norm(lg[j])), lw=1.2, zorder=4)
            ln._lgval = lg[j]
        if self.state is not None:                  # current target curve
            s = self.state
            w.axes[0].plot(s["foh"], s["cloh"], "k-", lw=2, zorder=5,
                           label="target")
            w.axes[1].plot(s["foh"] / s["cloh"], s["cloh"], "k-", lw=2,
                           zorder=5)
        self._plot_obs(w.axes[0], "foh", on_top=True, size=self._obs_size())
        self._plot_obs(w.axes[1], "fcl", on_top=True, size=self._obs_size())
        w.axes[0].set_xlabel("XF/XOH"); w.axes[0].set_ylabel("XCl/XOH")
        w.axes[1].set_xlabel("XF/XCl"); w.axes[1].set_ylabel("XCl/XOH")
        mname = self.ms_metric.get()
        w.axes[0].set_title(f"multistart runs coloured by log10({mname})",
                            fontsize=8)
        w.axes[0].legend(fontsize=6)
        for ax in w.axes:
            ax.grid(alpha=0.3)
        w.fig.tight_layout()            # BEFORE the colorbar (same order as
        # the embedded tab: tight_layout + make_axes colorbars don't mix)
        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        w._ms_cbar = w.fig.colorbar(sm, ax=w.axes[1], pad=0.02)
        w._ms_cbar.set_label(f"log$_{{10}}$ {mname}", fontsize=9)
        w._ms_cbar.ax.tick_params(labelsize=8)
        self._mark_topn_cbar(w._ms_cbar, lg, n_top_eff)
        w.canvas.draw_idle()
        self._note_autosave(w.autosave())

    def _note_autosave(self, paths):
        """Popout autosaves land in a session temp dir that is only packed
        INSIDE the .otproj zip (figures/) on project save — invisible on
        disk, which users read as a failed save (2026-08-06). Say where
        they went; Save figures / Save... write real on-disk copies."""
        if paths:
            self.status.config(
                text=f"popout autosaved -> {os.path.dirname(paths[0])} "
                     "(packed into the .otproj figures/ on project save; "
                     "use Save figures for an on-disk copy)")

    def _open_rmse_diag(self):
        """RMSE 判别图 window (spec M2-5, Lormand-style sensitivity panels):
        (i) sorted metric + histogram with the acceptance cutoff; (ii) metric vs
        each inverted parameter; (iii) Cl_init-F_init pair coloured by metric.
        Metric = the run objective (identical by construction)."""
        if not self.ms_results:
            messagebox.showinfo("Multistart", "run the multistart first")
            return
        from ot_figwin import FigureWindow
        import matplotlib as mpl
        import matplotlib.colors as mcolors
        rmses = np.array([r[0] for r in self.ms_results])
        X = np.array([r[1] for r in self.ms_results])
        try:
            mult = float(self.rmse_mult.get())
        except tk.TclError:
            mult = 5.0
        cutoff = rmses[0] * mult
        acc = rmses <= cutoff
        mname = self.ms_metric.get()
        w = FigureWindow(self.root, f"{mname} diagnostics", 3, 4,
                         figsize=(12.5, 8.2), figtype="rmse_diag",
                         autosave_dir=self.figures_dir())
        ax = w.axes
        # (i) sorted metric + histogram
        ax[0].semilogy(np.arange(1, len(rmses) + 1), rmses, ".", ms=3,
                       color=WONG["blue"])
        ax[0].axhline(cutoff, color=WONG["orange"], ls="--", lw=1)
        ax[0].set_xlabel("rank"); ax[0].set_ylabel(mname)
        ax[0].set_title(f"sorted runs (cutoff = {mult:g}x best)", fontsize=8)
        pos = rmses[rmses > 0]
        bins = (np.logspace(np.log10(pos.min()), np.log10(pos.max()), 25)
                if pos.size and pos.min() < pos.max() else 25)
        ax[1].hist(pos, bins=bins, color=WONG["blue"])
        ax[1].axvline(cutoff, color=WONG["orange"], ls="--", lw=1)
        ax[1].set_xscale("log")
        ax[1].set_xlabel(mname)
        ax[1].set_title("histogram", fontsize=8)
        # (ii) metric vs each parameter (sensitivity)
        for i, k in enumerate(PARAM_ORDER):
            a = ax[2 + i]
            a.semilogy(X[~acc, i], rmses[~acc], ".", ms=3, color="0.75")
            a.semilogy(X[acc, i], rmses[acc], ".", ms=4, color=WONG["blue"])
            a.axhline(cutoff, color=WONG["orange"], ls="--", lw=0.8)
            a.set_xlabel(k, fontsize=7)
            lo, hi = self.bvars[k][0].get(), self.bvars[k][1].get()
            a.set_xlim(float(lo), float(hi))
        # (iii) parameter pair coloured by metric
        pair = ax[11]
        lg = np.log10(np.maximum(rmses, 1e-12))
        sc = pair.scatter(X[:, 1], X[:, 2], c=lg, cmap="jet", s=10)
        pair.set_xlabel("Cl_init"); pair.set_ylabel("F_init")
        pair.set_title(f"pair view, colour = log10({mname})", fontsize=8)
        w.fig.colorbar(sc, ax=pair, fraction=0.05)
        for a in w.axes:
            a.grid(alpha=0.25)
            a.tick_params(labelsize=6)
        w.fig.suptitle(f"{mname} diagnostics — accepted {int(acc.sum())}"
                       f"/{len(rmses)}", fontsize=9)
        w.fig.tight_layout(rect=(0, 0, 1, 0.96))
        w.canvas.draw_idle()
        self._note_autosave(w.autosave())

    def _save_ms_figures(self):
        """Save the three embedded multistart panels (ratio fan x2 + metric
        histogram) as PNG 300 dpi + PDF — combined and per panel — into a
        visible timestamped folder exports/YYYYMMDD_HHMMSS/ under
        visible_out_base(), same convention as the Explore-tab export
        bundle, then open the folder (user request 2026-08-06)."""
        if not self.ms_results:
            messagebox.showinfo("Multistart", "run the multistart first")
            return
        base = visible_out_base()
        out = os.path.join(base, "exports",
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            os.makedirs(out, exist_ok=True)
            # the on-screen caption/headline live in Tk labels below the
            # canvas; carry them into the combined export as a footnote
            cap = "\n".join(t for t in (self.ms_head.cget("text"),
                                        self.ms_caption.cget("text")) if t)
            note = self.fig4.text(0.5, -0.02, cap, ha="center", va="top",
                                  fontsize=8, color="#555555")
            try:
                self.fig4.savefig(os.path.join(out, "ms_bundle.png"),
                                  dpi=300, bbox_inches="tight")
                self.fig4.savefig(os.path.join(out, "ms_bundle.pdf"),
                                  bbox_inches="tight")
            finally:
                note.remove()
            self.cv4.draw()             # renderer for per-panel tight boxes
            rend = self.cv4.get_renderer()
            for ax, stem in zip(self.ax4, ("fan_xfoh", "fan_xfxcl",
                                           "metric_hist")):
                bb = ax.get_tightbbox(rend).transformed(
                    self.fig4.dpi_scale_trans.inverted()).expanded(1.04, 1.06)
                self.fig4.savefig(os.path.join(out, f"ms_{stem}.png"),
                                  dpi=300, bbox_inches=bb)
                self.fig4.savefig(os.path.join(out, f"ms_{stem}.pdf"),
                                  bbox_inches=bb)
        except Exception as e:
            messagebox.showerror("Save figures", str(e))
            return
        self.status.config(text="multistart figures (combined + 3 panels, "
                                f"PNG/PDF) -> {out}")
        try:
            os.startfile(out)           # Windows-only app (per README_EXE)
        except OSError:
            pass

    # ---------------- multistart ----------------
    def _ftol_value(self):
        """Safe-parse the tolerance entry (fallback = engine default 1e-10)."""
        try:
            v = float(self.ms_ftol.get())
            return v if v > 0 else 1e-10
        except Exception:
            return 1e-10

    def _center_bounds(self):
        p = self.current_params()
        for k in PARAM_ORDER:
            lo, hi = PARAM_RANGE[k]
            self.bvars[k][0].set(round(max(lo, p[k] * 0.7), 4))
            self.bvars[k][1].set(round(min(hi, p[k] * 1.3), 4))

    def _run_multistart(self):
        if self._ms_thread is not None and self._ms_thread.is_alive():
            messagebox.showinfo("Multistart", "a run is already in progress")
            return
        try:
            bounds = {k: (float(self.bvars[k][0].get()),
                          float(self.bvars[k][1].get())) for k in PARAM_ORDER}
            for k, (lo, hi) in bounds.items():
                if not lo < hi:
                    raise ValueError(f"{k}: lb must be < ub")
            ns = int(self.n_start.get())
            nj = max(1, int(self.n_jobs.get()))
            ftol = float(self.ms_ftol.get())
            if not ftol > 0:
                raise ValueError("tolerance (ftol) must be > 0")
            n_explore = max(0, int(self.explore_n.get()))
        except Exception as e:
            messagebox.showerror("Multistart", str(e))
            return
        # target curve = the CURRENT forward (Lormand: fit the hand-set target)
        # TODO(task 2.2): the multistart objective is inv.rmse_to_target --
        # RMSE to THIS target curve over XCl/XOH + XF/XOH (engine, Lormand
        # stage 2). It is intentionally NOT the displayed zir_misfit (median
        # normalized distance curve->ZIR scatter); keep both definitions apart.
        try:
            F_t, T_t = self._grid(200)
            st = compute_state(self.current_params(), float(self.kspec.get()),
                               F_t, T_t)
            F_fit, T_fit = self._grid(NUM_F_FIT)
        except Exception as e:            # e.g. invalid T-F formula entries
            messagebox.showerror("Multistart", f"target forward failed: {e}")
            return
        args = dict(F_obs=st["F"], tgt_cloh=st["cloh"], tgt_foh=st["foh"],
                    F_fit=F_fit, T_fit=T_fit, bounds=bounds, ns=ns, nj=nj,
                    k_spec=float(self.kspec.get()),
                    ftol=ftol, n_explore=n_explore,
                    metric=self.ms_metric.get())
        # context recorded into the project when the run completes
        self._run_ctx = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            note=self.run_note.get().strip(),
            ms_ftol=ftol, explore_n=n_explore, metric=self.ms_metric.get(),
            params=self.current_params(), k_spec=float(self.kspec.get()),
            const_T=float(self.const_T.get()), t_mode=self.t_mode.get(),
            tf_loaded=self.tf is not None, tf_formula=self._formula_values(),
            bounds={k: list(v) for k, v in bounds.items()},
            n_start=ns, rmse_mult=float(self.rmse_mult.get()))
        self.btn_run.config(state="disabled")
        self.pbar.config(value=0, maximum=ns)
        self._ms_t0 = time.time()
        self._ms_nj = min(nj, ns)
        per_cpu = load_fit_time()                  # measured on a previous run
        if per_cpu is not None:
            est = ns * per_cpu / self._ms_nj
            self.lbl_ms.config(text=f"running 0/{ns}  est ~{fmt_dur(est)}")
        else:
            self.lbl_ms.config(text=f"running 0/{ns}  (estimating...)")
        self._ms_thread = threading.Thread(target=self._ms_worker,
                                           args=(args,), daemon=True)
        self._ms_thread.start()
        self.root.after(200, self._ms_poll)

    def _ms_worker(self, a):
        """Background thread: fan the L-BFGS-B starts over a process pool,
        reporting per-start progress. Reuses inv._worker_init / inv._fit_one
        (same code path as the engine; deterministic in inv.SEED)."""
        try:
            lb = np.array([a["bounds"][k][0] for k in PARAM_ORDER])
            ub = np.array([a["bounds"][k][1] for k in PARAM_ORDER])
            bounds_list = list(zip(lb, ub))
            rng = np.random.default_rng(inv.SEED)
            starts = [lb + rng.random(len(lb)) * (ub - lb) for _ in range(a["ns"])]
            ftol = a.get("ftol")
            lsopts = (None if ftol is None or ftol <= 1e-10
                      else dict(inv._LSOPTS, ftol=ftol))
            init_args = (a["F_fit"], a["T_fit"], a["F_obs"], a["tgt_cloh"],
                         a["tgt_foh"], bounds_list, a["k_spec"], 0.0, lsopts,
                         a.get("metric", "rmse"))
            results = []
            if a["nj"] == 1:
                inv._worker_init(*init_args)
                for i, x0 in enumerate(starts):
                    results.append(inv._fit_one(x0))
                    self._ms_queue.put(("prog", i + 1))
            else:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=min(a["nj"], a["ns"]),
                                         initializer=inv._worker_init,
                                         initargs=init_args) as ex:
                    futs = [ex.submit(inv._fit_one, x0) for x0 in starts]
                    for i, fut in enumerate(as_completed(futs)):
                        results.append(fut.result())
                        self._ms_queue.put(("prog", i + 1))
            results.sort(key=lambda t: t[0])
            # explore draws: forward-only random bounds samples (no optimiser) —
            # the prior-predictive curve space the BOUNDS allow. Same rng stream
            # (continues after the starts), so deterministic in inv.SEED.
            explore = []
            for _ in range(int(a.get("n_explore", 0))):
                x0 = lb + rng.random(len(lb)) * (ub - lb)
                r = inv.rmse_to_target(x0, a["F_fit"], a["T_fit"], a["F_obs"],
                                       a["tgt_cloh"], a["tgt_foh"],
                                       a["k_spec"], 0.0,
                                       metric=a.get("metric", "rmse"))
                explore.append((float(r), x0))
            self._ms_queue.put(("done", (results, explore)))
        except Exception as e:
            self._ms_queue.put(("err", str(e)))

    def _ms_poll(self):
        alive = self._ms_thread is not None and self._ms_thread.is_alive()
        try:
            while True:
                kind, payload = self._ms_queue.get_nowait()
                if kind == "prog":
                    self.pbar.config(value=payload)
                    total = int(self.pbar["maximum"])
                    elapsed = time.time() - self._ms_t0
                    txt = f"running {payload}/{total}  {fmt_dur(elapsed)}"
                    if payload > 0:                # extrapolate from throughput
                        eta = elapsed / payload * (total - payload)
                        txt += f"  ~{fmt_dur(eta)} left"
                    self.lbl_ms.config(text=txt)
                elif kind == "done":
                    results, explore = payload
                    elapsed = time.time() - self._ms_t0
                    # remember per-start CPU time for next run's upfront estimate
                    save_fit_time(elapsed * self._ms_nj / max(1, len(results)))
                    if self._env_cur is not None:   # envelope chain (3 targets)
                        self._env_results[self._env_cur] = results
                        if self._env_pending:
                            self._env_start_next()
                        else:
                            self._env_cur = None
                            self.btn_run.config(state="normal")
                            self.btn_env.config(state="normal")
                            self.lbl_ms.config(text="envelope done")
                            self._show_env_results()
                    else:
                        self.ms_results = results
                        self.ms_explore = explore
                        self.lbl_ms.config(text=f"done ({len(results)} fits, "
                                                f"{fmt_dur(elapsed)})")
                        self.btn_run.config(state="normal")
                        self._show_ms_results()
                        self._record_run(results)
                elif kind == "err":
                    self.lbl_ms.config(text="FAILED")
                    self.btn_run.config(state="normal")
                    self.btn_env.config(state="normal")
                    self._env_cur = None
                    self._env_pending = []
                    messagebox.showerror("Multistart", payload)
        except queue.Empty:
            pass
        if alive or (self._ms_thread is not None and self._ms_thread.is_alive()):
            self.root.after(200, self._ms_poll)

    def _show_ms_results(self):
        res = self.ms_results
        rmses = np.array([r[0] for r in res])
        X = np.array([r[1] for r in res])
        best = rmses[0]
        cutoff = best * float(self.rmse_mult.get())
        acc = rmses <= cutoff
        bounds = {k: (float(self.bvars[k][0].get()),
                      float(self.bvars[k][1].get())) for k in PARAM_ORDER}

        mname = self.ms_metric.get() or "rmse"
        self.ms_head.config(text=f"best {mname} = {best:.4g}   accepted "
                                 f"(<= {cutoff:.4g}): {int(acc.sum())}"
                                 f"/{len(rmses)}")
        rows, tags = [], []
        for i, k in enumerate(PARAM_ORDER):
            lo, hi = bounds[k]
            col = X[acc, i]
            span = hi - lo
            rail = ("LB" if np.mean(col <= lo + 0.02 * span) > 0.3 else
                    ("UB" if np.mean(col >= hi - 0.02 * span) > 0.3 else "-"))
            rows.append((k, f"{lo:.3g}", f"{hi:.3g}", f"{col.min():.4g}",
                         f"{np.median(col):.4g}", f"{col.max():.4g}", rail))
            tags.append("rail" if rail != "-" else "")
        self._fill_ms_table(
            ("param", "lb", "ub", "min", "med", "max", "rail"),
            (80, 48, 48, 58, 58, 58, 34), rows, tags)
        h2o, cl, fw = X[acc, 0], X[acc, 1], X[acc, 2]
        notes = [("accepted-family melt initial volatiles:  "
                  f"H2O {h2o.min():.2f}..{h2o.max():.2f} "
                  f"(med {np.median(h2o):.2f})   "
                  f"Cl {cl.min():.3f}..{cl.max():.3f} "
                  f"(med {np.median(cl):.3f})   "
                  f"F {fw.min():.3f}..{fw.max():.3f} "
                  f"(med {np.median(fw):.3f}) wt%"),
                 ("NB: family width is conditional on the hand-set target "
                  "curve + these BOUNDS (Lormand two-stage), NOT data error.")]
        try:
            _ftol = float(self.ms_ftol.get())
        except Exception:
            _ftol = 1e-10
        if _ftol > 1e-8:
            notes.append(f"! ftol={_ftol:g} LOOSE (engine default 1e-10): "
                         "runs stop early -> the fan/family width is "
                         "DISPLAY/exploration only, NOT for reporting.")
        if self.ms_explore:
            notes.append(f"({len(self.ms_explore)} explore draws underlaid — "
                         "no optimisation)")
        self.ms_notes.config(text="\n".join(notes))

        # figure: curve bundle (2 ratio panels) + RMSE histogram.
        # Lormand-MATLAB style (per user 2026-07-04, GUI readme Fig 7): show ALL
        # successful multistart runs — not just the accepted family — coloured
        # by log10(RMSE) (jet + colorbar), worst drawn first so the good (dark)
        # curves sit on top. The high-RMSE fan is what envelopes the data; the
        # accepted family always hugs the target curve by construction.
        import matplotlib as mpl
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        self._ms_axes_on()                          # v2 A4
        F_fit, T_fit = self._grid(NUM_F_FIT)
        acc_idx = np.where(acc)[0]                  # results are RMSE-sorted
        all_idx = np.arange(len(rmses))
        if len(all_idx) > 300:                      # cap forwards for speed,
            all_idx = all_idx[np.unique(            # spread-sampled over ALL runs
                np.linspace(0, len(all_idx) - 1, 300).astype(int))]
        for ax in self.ax4:
            ax.clear()
        try:
            n_top = max(0, int(self.topn.get()))
        except tk.TclError:
            n_top = 5
        self.ms_overlay = []                        # top-N for Eye-fit (4.3)
        for j in acc_idx[:n_top]:
            p = inv.coeff_to_params(X[j], 0.0)
            self.ms_overlay.append(inv.forward_ratios(p, F_fit, T_fit,
                                                      float(self.kspec.get())))
        exp_rm = np.array([e[0] for e in self.ms_explore]) if self.ms_explore \
            else np.array([])
        lg = np.log10(np.maximum(rmses, 1e-12))
        lg_all = (np.concatenate([lg, np.log10(np.maximum(exp_rm, 1e-12))])
                  if exp_rm.size else lg)
        norm = mcolors.Normalize(vmin=lg_all.min(), vmax=lg_all.max())
        self._ms_norm = norm            # kept for in-place recolouring
        # user-selected fan colormap (cm.get_cmap was removed in mpl 3.9)
        cmap = mpl.colormaps[self.ms_cmap.get()]
        # explore draws first (bottom layer): the raw curve space of the BOUNDS
        exp_idx = np.arange(len(self.ms_explore))
        if len(exp_idx) > 300:
            exp_idx = exp_idx[np.unique(
                np.linspace(0, len(exp_idx) - 1, 300).astype(int))]
        for j in exp_idx:
            rj, xj = self.ms_explore[j]
            p = inv.coeff_to_params(np.asarray(xj), 0.0)
            cloh, foh = inv.forward_ratios(p, F_fit, T_fit,
                                           float(self.kspec.get()))
            lgv = np.log10(max(rj, 1e-12))
            for a, xx in ((self.ax4[0], foh), (self.ax4[1], foh / cloh)):
                ln, = a.plot(xx, cloh, "-", color=cmap(norm(lgv)),
                             alpha=0.22, lw=0.5, zorder=1.0)
                ln._lgval = lgv         # recolourable (2026-08-04 cmap option)
        for j in all_idx[::-1]:                     # worst first, best on top
            p = inv.coeff_to_params(X[j], 0.0)
            cloh, foh = inv.forward_ratios(p, F_fit, T_fit,
                                           float(self.kspec.get()))
            for a, xx in ((self.ax4[0], foh), (self.ax4[1], foh / cloh)):
                ln, = a.plot(xx, cloh, "-", color=cmap(norm(lg[j])),
                             alpha=0.5, lw=0.7)
                ln._lgval = lg[j]
        self._plot_obs(self.ax4[0], "foh", on_top=True,
                       size=self._obs_size())
        self._plot_obs(self.ax4[1], "fcl", on_top=True,
                       size=self._obs_size())
        self.ax4[0].set_xlabel("XF/XOH"); self.ax4[0].set_ylabel("XCl/XOH")
        self.ax4[1].set_xlabel("XF/XCl"); self.ax4[1].set_ylabel("XCl/XOH")
        # panel descriptions live in the caption BELOW the figure
        # (2026-08-04 — in-axes titles stole vertical plot space)
        cap = f"all multistart runs ({len(all_idx)}/{len(rmses)} shown)"
        if self.ms_explore:
            cap += f" + {len(exp_idx)} explore draws"
        # RMSE histogram on a LOG x-axis (per user 2026-07-04): the family piles
        # up within ~2x of the best RMSE, so linear bins collapse into one bar.
        pos = rmses[rmses > 0]
        if pos.size and pos.min() < pos.max():
            bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 30)
        else:
            bins = 30
        self.ax4[2].hist(pos, bins=bins, color=WONG["blue"])
        self.ax4[2].axvline(cutoff, color=WONG["orange"], ls="--", lw=1)
        self.ax4[2].set_xscale("log")
        # v2 B3: only major (decade) tick labels, slightly rotated — the
        # minor 2x/3x/6x labels used to overlap into an unreadable pile
        self.ax4[2].tick_params(axis="x", which="minor", labelbottom=False)
        self.ax4[2].tick_params(axis="x", labelrotation=20)
        self.ax4[2].set_xlabel(f"{mname} (log scale)")
        self.ms_caption.config(
            text=cap + f"    |    {mname} histogram, acceptance cutoff dashed")
        for ax in self.ax4:
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        if getattr(self, "_ms_cbar", None) is None:
            # first display: settle the 3-panel layout, THEN carve the colorbar
            # axes ONCE (tight_layout is incompatible with make_axes colorbars,
            # and re-creating the bar would shrink ax4[1] every refresh)
            self.fig4.tight_layout()
            self._ms_cbar = self.fig4.colorbar(sm, ax=self.ax4[1], pad=0.02)
        else:
            self._ms_cbar.update_normal(sm)
        self._ms_cbar.set_label(f"log$_{{10}}$ {mname}", fontsize=9)
        self._ms_cbar.ax.tick_params(labelsize=8)
        # top-N threshold on the colorbar (user 2026-08-06): ranks over ALL
        # sorted runs, same definition as the popout's top-N curves
        self._mark_topn_cbar(self._ms_cbar, lg, min(n_top, len(rmses)))
        self.cv4.draw_idle()
        if self.state is not None:                  # show top-N on Eye-fit (4.3)
            self._draw_eyefit()

    # ------------- envelope: 26.6.24 multi-target honest uncertainty -------------
    # A single Lormand target gives a family that is parameter trade-off only.
    # The honest melt-volatile range = UNION of the accepted families of THREE
    # eye-validated targets bracketing the observed cloud (ot_target_envelope.py).
    def _env_set(self, name):
        """Save the CURRENT sliders as the low/central/high bracketing target."""
        self.env_targets[name] = self.current_params()
        self._env_refresh_lbls()

    def _env_refresh_lbls(self):
        if not getattr(self, "env_lbls", None):
            return
        for n, lbl in self.env_lbls.items():
            t = self.env_targets.get(n)
            lbl.config(text="-" if t is None else
                       f"H2O {t['H2O_init']:.2f} Cl {t['Cl_init']:.3f} "
                       f"F {t['F_init']:.3f}")

    def _env_suggest(self):
        """Prefill low/high by scaling (F_init, Cl_init) off the central target
        with the 26.6.24 factors — a STARTING point; eye-validate + retune."""
        if self.env_targets["central"] is None:
            self.env_targets["central"] = self.current_params()
        c = self.env_targets["central"]
        for n, (fF, fCl) in ENV_SCALE.items():
            d = dict(c)
            d["F_init"] = c["F_init"] * fF
            d["Cl_init"] = c["Cl_init"] * fCl
            self.env_targets[n] = d
        self._env_refresh_lbls()
        self._env_preview()

    def _env_preview(self):
        """Forward the saved targets and overlay them on the Eye-fit panels —
        the eye-validation step: retune until the band wraps the data."""
        if all(v is None for v in self.env_targets.values()):
            messagebox.showinfo("Envelope", "set (or suggest) targets first")
            return
        F_t, T_t = self._grid(200)
        self.env_curves = {}
        for n, t in self.env_targets.items():
            if t is not None:
                st = compute_state(t, float(self.kspec.get()), F_t, T_t)
                self.env_curves[n] = (st["cloh"], st["foh"])
        if self.state is not None:
            self._draw_eyefit()
        self.nb.select(0)                       # jump to Eye-fit to inspect

    def _run_envelope(self):
        if self._ms_thread is not None and self._ms_thread.is_alive():
            messagebox.showinfo("Multistart", "a run is already in progress")
            return
        missing = [n for n, t in self.env_targets.items() if t is None]
        if missing:
            messagebox.showerror("Envelope", "targets not set: " + ", ".join(missing)
                                 + "\n(use the sliders + 'set ...', or 'suggest')")
            return
        try:
            bounds = {k: (float(self.bvars[k][0].get()),
                          float(self.bvars[k][1].get())) for k in PARAM_ORDER}
            for k, (lo, hi) in bounds.items():
                if not lo < hi:
                    raise ValueError(f"{k}: lb must be < ub")
            ns = int(self.n_start.get())
            nj = max(1, int(self.n_jobs.get()))
        except Exception as e:
            messagebox.showerror("Envelope", str(e))
            return
        # WIDE-bounds requirement: the envelope must not be clipped, so the
        # bounds must contain every bracketing target (ot_target_envelope.py).
        viol = [f"  {n}: {k}={t[k]:.4g} outside [{bounds[k][0]:g}, {bounds[k][1]:g}]"
                for n, t in self.env_targets.items()
                for k in PARAM_ORDER if not bounds[k][0] <= t[k] <= bounds[k][1]]
        if viol:
            messagebox.showerror(
                "Envelope", "widen BOUNDS to contain all 3 targets:\n"
                + "\n".join(viol[:12]) + ("\n..." if len(viol) > 12 else ""))
            return
        self._env_results = {}
        self._env_pending = [n for n in ("low", "central", "high")]
        self._env_bounds, self._env_ns, self._env_nj = bounds, ns, nj
        self.btn_run.config(state="disabled")
        self.btn_env.config(state="disabled")
        self._env_start_next()
        self.root.after(200, self._ms_poll)

    def _env_start_next(self):
        name = self._env_pending.pop(0)
        self._env_cur = name
        tgt = self.env_targets[name]
        F_t, T_t = self._grid(200)
        st = compute_state(tgt, float(self.kspec.get()), F_t, T_t)
        F_fit, T_fit = self._grid(NUM_F_FIT)
        try:
            ftol = float(self.ms_ftol.get())
        except Exception:
            ftol = None
        args = dict(F_obs=st["F"], tgt_cloh=st["cloh"], tgt_foh=st["foh"],
                    F_fit=F_fit, T_fit=T_fit, bounds=self._env_bounds,
                    ns=self._env_ns, nj=self._env_nj,
                    k_spec=float(self.kspec.get()),
                    ftol=ftol, n_explore=0,     # explore overlay: single-run only
                    metric=self.ms_metric.get())
        self.pbar.config(value=0, maximum=self._env_ns)
        self._ms_t0 = time.time()
        self._ms_nj = min(self._env_nj, self._env_ns)
        done = 3 - 1 - len(self._env_pending)
        self.lbl_ms.config(text=f"envelope [{name}] ({done}/3 done)")
        self._ms_thread = threading.Thread(target=self._ms_worker,
                                           args=(args,), daemon=True)
        self._ms_thread.start()

    def _show_env_results(self):
        try:
            mult = float(self.rmse_mult.get())
        except tk.TclError:
            mult = 5.0
        F_fit, T_fit = self._grid(NUM_F_FIT)
        self._ms_axes_on()                          # v2 A4
        for ax in self.ax4:
            ax.clear()
        self.ms_head.config(
            text="ENVELOPE (26.6.24): union of the 3 accepted families "
                 f"— accept <= {mult:g} x best PER TARGET")
        rows = []
        fams = []                               # (name, accepted X) in run order
        for n in ("low", "central", "high"):
            res = self._env_results.get(n)
            if not res:
                continue
            rm = np.array([r[0] for r in res])
            X = np.array([r[1] for r in res])
            acc = rm <= mult * rm[0]
            Xa = X[acc]
            fams.append((n, Xa))
            h, c, f = Xa[:, 0], Xa[:, 1], Xa[:, 2]
            rows.append((n, f"{rm[0]:.4g}", f"{len(Xa)}/{len(rm)}",
                         f"{h.min():.2f}-{h.max():.2f}",
                         f"{c.min():.3f}-{c.max():.3f}",
                         f"{f.min():.3f}-{f.max():.3f}"))
            ai = np.where(acc)[0]               # bundle: <=40 spread-sampled
            if len(ai) > 40:
                ai = ai[np.unique(np.linspace(0, len(ai) - 1, 40).astype(int))]
            for j in ai:
                p = inv.coeff_to_params(X[j], 0.0)
                cloh, foh = inv.forward_ratios(p, F_fit, T_fit,
                                               float(self.kspec.get()))
                self.ax4[0].plot(foh, cloh, "-", color=ENV_COL[n],
                                 alpha=0.18, lw=0.7)
                self.ax4[1].plot(foh / cloh, cloh, "-", color=ENV_COL[n],
                                 alpha=0.18, lw=0.7)
            pos = rm[rm > 0]
            if pos.size and pos.min() < pos.max():
                bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 24)
            else:
                bins = 24
            self.ax4[2].hist(pos, bins=bins, color=ENV_COL[n], alpha=0.55,
                             label=n)
        U = np.vstack([Xa for _, Xa in fams])
        uni = []
        for i, nm, fmt in ((0, "H2O", "{:.2f}"), (1, "Cl", "{:.3f}"),
                           (2, "F", "{:.3f}")):
            col = U[:, i]
            uni.append(f"{nm} {fmt.format(col.min())}.."
                       f"{fmt.format(col.max())} "
                       f"(med {fmt.format(np.median(col))})")
        rows.append(("UNION", "", f"n={len(U)}", uni[0].split(" ", 1)[1],
                     uni[1].split(" ", 1)[1], uni[2].split(" ", 1)[1]))
        self._fill_ms_table(
            ("target", "best", "accepted", "H2O wt%", "Cl wt%", "F wt%"),
            (56, 56, 60, 88, 88, 88), rows,
            [""] * (len(rows) - 1) + ["rail"])
        lines = ["UNION — honest melt-volatile range: " + "  ".join(uni) + " wt%",
                 "NB: each family is target-conditional (trade-off only); "
                 "the UNION across the brackets is the range to report."]
        out_dir = os.path.dirname(self.obs_path) if self.obs_path else ""
        if not out_dir or _temp_rooted(out_dir):
            # an obs CSV restored from a .otproj lives in a %TEMP%\otproj_*
            # extraction dir -- the union CSVs must not land there
            out_dir = visible_out_base()
        try:
            for n, res in self._env_results.items():
                inv.write_multioutput(res, os.path.join(out_dir,
                                                        f"gui_env_{n}.csv"))
            with open(os.path.join(out_dir, "gui_env_union.csv"), "w",
                      newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["target"] + PARAM_ORDER)
                for n, Xa in fams:
                    for row in Xa:
                        w.writerow([n] + [f"{v:.6g}" for v in row])
            lines.append("[exported] gui_env_{low,central,high,union}.csv -> "
                         + out_dir)
        except Exception as e:
            lines.append(f"[export FAILED] {e}")
        self.ms_notes.config(text="\n".join(lines))
        self._plot_obs(self.ax4[0], "foh", on_top=True,
                       size=self._obs_size())
        self._plot_obs(self.ax4[1], "fcl", on_top=True,
                       size=self._obs_size())
        self.ax4[0].set_xlabel("XF/XOH"); self.ax4[0].set_ylabel("XCl/XOH")
        self.ax4[1].set_xlabel("XF/XCl"); self.ax4[1].set_ylabel("XCl/XOH")
        self.ms_caption.config(
            text="envelope accepted families (blue=low black=central "
                 f"orange=high)    |    {self.ms_metric.get()} per target")
        self.ax4[2].set_xscale("log")
        self.ax4[2].tick_params(axis="x", which="minor", labelbottom=False)
        self.ax4[2].tick_params(axis="x", labelrotation=20)
        self.ax4[2].set_xlabel(f"{self.ms_metric.get()} (log scale)")
        self.ax4[2].legend(fontsize=8)
        for ax in self.ax4:
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        self.cv4.draw_idle()

    def _export_multioutput(self):
        if not self.ms_results:
            messagebox.showinfo("Export", "run the multistart first")
            return
        p = filedialog.asksaveasfilename(
            title="Export Multioutput CSV", defaultextension=".csv",
            initialdir=os.environ.get("OT_OUT", os.getcwd()),
            initialfile="gui_multistart.csv")
        if p:
            inv.write_multioutput(self.ms_results, p)
            self.status.config(text=f"exported {os.path.basename(p)}")

    def _apply_best(self):
        if not self.ms_results:
            messagebox.showinfo("Apply", "run the multistart first")
            return
        for k, v in zip(PARAM_ORDER, self.ms_results[0][1]):
            self.pvars[k].set(round(float(v), 5))
        self._schedule_recompute()
        self.nb.select(0)

    # ---------------- run overlays (task 4.2) ----------------
    def _add_overlay(self):
        sel = self.runs_list.curselection()
        if not sel:
            messagebox.showinfo("Overlay", "select a run in the list first")
            return
        self.overlay_idx.add(sel[0])
        self._refresh_runs_list()
        if self.state is not None:
            self._draw_eyefit()

    def _clear_overlays(self):
        self.overlay_idx.clear()
        self.ms_overlay = []                    # also drops the top-N (4.3)
        self.env_curves = {}                    # and the envelope preview
        self._refresh_runs_list()
        if self.state is not None:
            self._draw_eyefit()

    # ---------------- project: record / save / load / replay ----------------
    def _snapshot_run(self):
        """Save the current parameter set into the project run history as a
        browsable run without a multistart family (task 4.2)."""
        rec = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            note=self.run_note.get().strip() or "snapshot",
            params=self.current_params(), k_spec=float(self.kspec.get()),
            const_T=float(self.const_T.get()), t_mode=self.t_mode.get(),
            tf_loaded=self.tf is not None, tf_formula=self._formula_values(),
            bounds={k: [float(self.bvars[k][0].get()),
                        float(self.bvars[k][1].get())] for k in PARAM_ORDER},
            n_start=0, rmse_mult=float(self.rmse_mult.get()),
            best_rmse=None, misfit=self.misfit, results=[])
        self.proj["runs"].append(rec)
        self._refresh_runs_list()
        if self.proj["path"]:                   # same autosave rule as runs
            try:
                self._write_project(self.proj["path"])
            except Exception as e:
                messagebox.showwarning("Project autosave", str(e))
        self.status.config(text=f"snapshot recorded as run "
                                f"#{len(self.proj['runs'])}")

    def _record_run(self, results):
        rec = dict(self._run_ctx)
        rec["best_rmse"] = float(results[0][0])
        rec["misfit"] = self.misfit
        rec["results"] = [(float(r), list(map(float, x))) for r, x in results]
        self.proj["runs"].append(rec)
        self._refresh_runs_list()
        if self.proj["path"]:                      # autosave into the open project
            try:
                self._write_project(self.proj["path"])
                self.status.config(text=f"run #{len(self.proj['runs'])} saved to "
                                        f"{os.path.basename(self.proj['path'])}")
            except Exception as e:
                messagebox.showwarning("Project autosave", str(e))

    def _refresh_runs_list(self):
        self.runs_list.delete(0, "end")
        for i, r in enumerate(self.proj["runs"]):
            note = f" {r['note']}" if r.get("note") else ""
            rm = r.get("best_rmse")
            head = f"n={r['n_start']} rmse={rm:.3g}" if rm is not None \
                else "snapshot"
            mf = r.get("misfit")
            mftxt = f" mf={mf:.3g}" if mf is not None else ""
            ov = "+" if i in self.overlay_idx else " "   # overlaid marker
            self.runs_list.insert(
                "end", f"{ov}#{i+1} {r['time']} {head}{mftxt}{note}")
        if self.proj["runs"]:
            self.runs_list.selection_clear(0, "end")
            self.runs_list.selection_set("end")
        ptxt = ((os.path.basename(self.proj["path"])
                 if self.proj["path"] else "(unsaved project)")
                + f"  [{len(self.proj['runs'])} runs]")
        self.lbl_proj.config(text=ptxt)
        self.sb_proj.config(text=ptxt)          # statusbar mirror (v2 C4)

    def _load_run(self):
        """Full replay of a recorded run: params + K_spec + bounds + settings
        are restored, then its stored family is shown."""
        sel = self.runs_list.curselection()
        if not sel:
            messagebox.showinfo("Runs", "select a run in the list first")
            return
        r = self.proj["runs"][sel[0]]
        for k in PARAM_ORDER:
            self.pvars[k].set(r["params"][k])
        self.kspec.set(r["k_spec"])
        self.const_T.set(r.get("const_T", 800.0))
        fm = r.get("tf_formula")                # pre-formula runs have none
        if fm:
            self.fml_Ti.set(fm.get("T_initial", 1000.0))
            self.fml_Tf.set(fm.get("T_final", 750.0))
            self.fml_b.set(fm.get("b", 2.0))
            self._draw_fml_preview()
        tm = r.get("t_mode")                    # pre-5.1 runs have no t_mode
        if tm is None:
            tm = "path" if r.get("tf_loaded") else "const"
        if tm == "path" and self.tf is None:
            tm = "const"
        self.t_mode.set(tm)
        self._update_tmode_widgets()
        for k, (lo, hi) in r["bounds"].items():
            self.bvars[k][0].set(lo)
            self.bvars[k][1].set(hi)
        self.n_start.set(r["n_start"])
        self.rmse_mult.set(r.get("rmse_mult", 5.0))
        self.run_note.set(r.get("note", ""))
        self.ms_ftol.set(f"{r.get('ms_ftol', 1e-10):g}")
        self.explore_n.set(int(r.get("explore_n", 0)))
        self.ms_metric.set(r.get("metric", "rmse"))   # pre-metric runs = rmse
        self.ms_explore = []                    # explore draws are not stored
        res = r["results"]
        self.ms_results = [(rm, np.array(x)) for rm, x in res] if res else None
        self._recompute()
        if self.ms_results:                     # snapshots carry no family
            self._show_ms_results()
            self.lbl_ms.config(text=f"replayed run #{sel[0]+1}")
        else:
            self.lbl_ms.config(text=f"loaded snapshot run #{sel[0]+1}")

    def _project_manifest(self):
        return dict(
            version=PROJ_VERSION, saved=datetime.now().isoformat(timespec="seconds"),
            params=self.current_params(), k_spec=float(self.kspec.get()),
            const_T=float(self.const_T.get()), t_mode=self.t_mode.get(),
            tf_formula=self._formula_values(),
            ui=dict(logx=bool(self.logx.get()),
                    collapsed=[t for t, g in self.groups.items()
                               if g.collapsed]),
            bounds={k: [float(self.bvars[k][0].get()),
                        float(self.bvars[k][1].get())] for k in PARAM_ORDER},
            n_start=int(self.n_start.get()), n_jobs=int(self.n_jobs.get()),
            rmse_mult=float(self.rmse_mult.get()),
            env_targets=self.env_targets,       # optional key; version stays 1
            ms_ftol=self._ftol_value(), explore_n=int(self.explore_n.get()),
            ms_metric=self.ms_metric.get(),
            m1=(self.m1.get_state()
                if getattr(self, "m1", None) and self.m1.raw is not None
                else None),
            smelt=(self.smelt.get_state()          # optional key, version 1
                   if getattr(self, "smelt", None) else None),
            obs=(dict(orig_path=self.obs_path,
                      embedded="data/" + os.path.basename(self.obs_path))
                 if self.obs_path else None),
            tf=(dict(orig_path=self.tf_path,
                     embedded="data/" + os.path.basename(self.tf_path))
                if self.tf_path else None),
            runs=[{k: v for k, v in r.items()
                   if k != "results" and not k.startswith("_")}
                  | {"csv": f"runs/run_{i+1:03d}.csv"}
                  for i, r in enumerate(self.proj["runs"])],
        )

    def _write_project(self, path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json",
                       json.dumps(self._project_manifest(), indent=1))
            for p in (self.obs_path, self.tf_path):     # embed data copies
                if p and os.path.exists(p):
                    z.write(p, "data/" + os.path.basename(p))
            for i, r in enumerate(self.proj["runs"]):
                z.writestr(f"runs/run_{i+1:03d}.csv",
                           results_to_csv_text(r["results"]))
            # M1 apatite data + auto-saved figures (P1 spec M0-3/M3-2)
            if getattr(self, "m1", None) and self.m1.raw is not None:
                z.writestr("data/apatite_raw.csv", self.m1.raw_csv_text())
                if self.m1.calc is not None:
                    z.writestr("data/apatite_calc.csv",
                               self.m1.calc_with_groups().to_csv(index=False))
            if self._fig_dir and os.path.isdir(self._fig_dir):
                for fn in sorted(os.listdir(self._fig_dir)):
                    z.write(os.path.join(self._fig_dir, fn), "figures/" + fn)

    def _save_project(self, ask=False):
        path = self.proj["path"]
        if ask or not path:
            path = filedialog.asksaveasfilename(
                title="Save project", defaultextension=PROJ_EXT,
                initialdir=os.environ.get("OT_OUT", os.getcwd()),
                filetypes=[("OT apatite project", "*" + PROJ_EXT)])
            if not path:
                return
        try:
            self._write_project(path)
        except Exception as e:
            messagebox.showerror("Save project", str(e))
            return
        self.proj["path"] = path
        push_recent(path)
        self._refresh_runs_list()
        try:                             # clean save supersedes the autosave
            os.remove(self._autosave_path(path))
        except OSError:
            pass
        self.status.config(text=f"project saved: {os.path.basename(path)}")

    def _open_project_dialog(self):
        path = filedialog.askopenfilename(
            title="Open project", filetypes=[("OT apatite project", "*" + PROJ_EXT)])
        if path:
            self.open_project(path)

    def open_project(self, path):
        # M0-4: a newer autosave (interrupted session) may be offered instead;
        # the SAVE target stays the original project path either way.
        src = (path if path.endswith(".autosave~")
               else self._maybe_recover_autosave(path))
        try:
            with zipfile.ZipFile(src) as z:
                man = json.loads(z.read("manifest.json").decode("utf-8"))
                if man.get("version", 0) > PROJ_VERSION:
                    raise ValueError("project written by a newer GUI version")
                tmp = tempfile.mkdtemp(prefix="otproj_")
                runs = []
                for rman in man.get("runs", []):
                    res = results_from_csv_text(z.read(rman["csv"]).decode("utf-8"))
                    rec = {k: v for k, v in rman.items() if k != "csv"}
                    rec["results"] = [(r, list(x)) for r, x in res]
                    runs.append(rec)
                data_paths = {}
                for key in ("obs", "tf"):
                    ent = man.get(key)
                    if ent:
                        orig = ent.get("orig_path")
                        if orig and os.path.exists(orig):
                            data_paths[key] = orig       # prefer the original file
                        else:
                            z.extract(ent["embedded"], tmp)
                            data_paths[key] = os.path.join(tmp, ent["embedded"])
                names = z.namelist()
                m1_raw = (z.read("data/apatite_raw.csv").decode("utf-8")
                          if "data/apatite_raw.csv" in names else None)
                for n in names:                      # restore figure history
                    if n.startswith("figures/") and not n.endswith("/"):
                        z.extract(n, tmp)
        except Exception as e:
            messagebox.showerror("Open project", str(e))
            return
        # apply state
        self._proj_tmp = tmp
        self.proj = dict(path=path, runs=runs)
        for k in PARAM_ORDER:
            self.pvars[k].set(man["params"][k])
        self.kspec.set(man["k_spec"])
        self.const_T.set(man.get("const_T", 800.0))
        fm = man.get("tf_formula")              # absent in pre-formula files
        if fm:
            self.fml_Ti.set(fm.get("T_initial", 1000.0))
            self.fml_Tf.set(fm.get("T_final", 750.0))
            self.fml_b.set(fm.get("b", 2.0))
        for k, (lo, hi) in man["bounds"].items():
            self.bvars[k][0].set(lo)
            self.bvars[k][1].set(hi)
        self.n_start.set(man.get("n_start", 100))
        self.n_jobs.set(man.get("n_jobs", self.n_jobs.get()))
        self.rmse_mult.set(man.get("rmse_mult", 5.0))
        self.env_targets = (man.get("env_targets")
                            or {"low": None, "central": None, "high": None})
        self.env_curves = {}
        self._env_refresh_lbls()
        self.ms_ftol.set(f"{man.get('ms_ftol', 1e-10):g}")
        self.explore_n.set(int(man.get("explore_n", 0)))
        self.ms_metric.set(man.get("ms_metric", "logrmse"))
        figs = os.path.join(tmp, "figures")
        if os.path.isdir(figs):
            self._fig_dir = figs
        st = man.get("m1")
        if st and getattr(self, "m1", None):
            try:
                self.m1.set_state(st, m1_raw)
            except Exception as e:
                messagebox.showwarning("M1 restore", str(e))
        st = man.get("smelt")             # absent in pre-melt-S projects
        if st and getattr(self, "smelt", None):
            try:
                self.smelt.set_state(st)
            except Exception as e:
                messagebox.showwarning("Melt S restore", str(e))
        if "obs" in data_paths:
            self._set_obs(data_paths["obs"])
        if "tf" in data_paths:
            self._set_tf(data_paths["tf"])       # sets t_mode="path"
        # restore UI state AFTER the data loads (pre-t_mode files: path if tf)
        tm = man.get("t_mode", "path" if self.tf is not None else "const")
        if tm == "path" and self.tf is None:
            tm = "const"
        self.t_mode.set(tm)
        if tm == "formula":
            self.fml_frame.set_collapsed(False)
        self._update_tmode_widgets()
        self._draw_fml_preview()
        ui = man.get("ui", {})
        self.logx.set(bool(ui.get("logx", False)))
        collapsed = set(ui.get("collapsed", []))
        for t, g in self.groups.items():
            g.set_collapsed(t in collapsed)
        self.overlay_idx.clear()
        push_recent(path)
        self._refresh_runs_list()
        if runs:                                   # show the latest run
            self.runs_list.selection_set("end")
            self._load_run()
        else:
            self._schedule_recompute()
        self.status.config(text=f"project opened: {os.path.basename(path)} "
                                f"({len(runs)} runs)")

    # ---------------- startup dialog ----------------
    def _open_sample_project(self, dlg):
        """v2 D1: demo session from the synthetic sample data, generated on
        demand (sample_data/ is gitignored — the GENERATOR is versioned)."""
        base = os.path.dirname(os.path.abspath(__file__))
        obs = os.path.join(base, "sample_data", "sample_obs.csv")
        tf = os.path.join(base, "sample_data", "sample_tf.csv")
        if not (os.path.exists(obs) and os.path.exists(tf)):
            try:
                import make_sample_project
                make_sample_project.main()
            except Exception as e:
                messagebox.showerror("Sample project",
                                     f"could not generate sample data: {e}")
                return
        dlg.destroy()
        self._set_obs(obs)
        self._set_tf(tf)
        self.status.config(text="sample session: synthetic demo data loaded "
                                "(obs + T-F curve)")

    def show_startup_dialog(self):
        """v2 D1: recents as name | folder | age (was two raw full paths in a
        Listbox), Enter/Escape/double-click bindings, sample-project button."""
        recent = load_recent()
        dlg = tk.Toplevel(self.root)
        dlg.title("Open project")
        dlg.transient(self.root)
        dlg.grab_set()
        ttk.Label(dlg, text="Recent projects:").pack(anchor="w", padx=10,
                                                     pady=(10, 2))

        def rel_age(path):
            try:
                mt = os.path.getmtime(path)
            except OSError:
                return "?"
            d = time.time() - mt
            if d < 3600:
                return f"{max(1, int(d // 60))} min ago"
            if d < 86400:
                return f"{int(d // 3600)} h ago"
            if d < 14 * 86400:
                return f"{int(d // 86400)} d ago"
            return datetime.fromtimestamp(mt).strftime("%Y-%m-%d")

        tv = ttk.Treeview(dlg, columns=("name", "folder", "modified"),
                          show="headings", selectmode="browse",
                          height=max(3, min(len(recent) or 1, MAX_RECENT)))
        for c, w, a in (("name", 190, "w"), ("folder", 330, "w"),
                        ("modified", 90, "e")):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor=a)
        for p in recent:
            tv.insert("", "end", values=(os.path.basename(p),
                                         os.path.dirname(p), rel_age(p)))
        tv.pack(padx=10, fill="x")
        if recent:
            first = tv.get_children()[0]
            tv.selection_set(first)
            tv.focus(first)
            tv.focus_set()

        def open_sel(*_):
            sel = tv.selection()
            if sel:
                p = recent[tv.index(sel[0])]
                dlg.destroy()
                self.open_project(p)

        def browse():
            dlg.destroy()
            self._open_project_dialog()

        tv.bind("<Double-Button-1>", open_sel)
        dlg.bind("<Return>", open_sel)
        dlg.bind("<Escape>", lambda *_: dlg.destroy())
        row = ttk.Frame(dlg)
        row.pack(pady=10)
        ttk.Button(row, text="Open selected", command=open_sel).pack(
            side="left", padx=4)
        ttk.Button(row, text="Browse...", command=browse).pack(side="left",
                                                               padx=4)
        ttk.Button(row, text="Open sample project",
                   command=lambda: self._open_sample_project(dlg)).pack(
            side="left", padx=4)
        ttk.Button(row, text="New blank session",
                   command=dlg.destroy).pack(side="left", padx=4)
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        # center over the main window
        dlg.update_idletasks()
        rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        dw, dh = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        dlg.geometry(f"+{max(0, rx + (rw - dw) // 2)}"
                     f"+{max(0, ry + (rh - dh) // 3)}")
        self.root.wait_window(dlg)


def setup_ui(root):
    """Window icon, unified fonts, and the sv_ttk theme (v2 B1/B2).
    One entry point so main() and any driver/test harness style the UI
    identically. Every step degrades gracefully."""
    try:                                        # custom window icon (task 5.2)
        icon = make_icon()
        root.iconphoto(True, icon)
        root._icon_ref = icon                   # keep alive
    except Exception:
        pass
    try:                                        # unified UI font, CJK-safe (5.3)
        from tkinter import font as tkfont
        fams = set(tkfont.families(root))
        fam = next((f for f in ("Microsoft YaHei UI", "Segoe UI")
                    if f in fams), None)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                     "TkHeadingFont"):
            f = tkfont.nametofont(name)
            f.configure(size=9, **({"family": fam} if fam else {}))
    except Exception:
        pass
    # v2 B1: Sun Valley ttk theme — OPTIONAL dependency (pip install sv-ttk).
    # Verified on this project's floor Tk 8.6.9 (Py3.9.0). Without it the
    # stock theme stays, so the other machine never breaks on a plain pull.
    try:
        import sv_ttk
        sv_ttk.set_theme("light")
    except Exception:
        pass


def main():
    global _SPLASH
    if not g08_validate():                          # refuse on a broken engine
        if _SPLASH is not None:
            _SPLASH.destroy()
        sys.exit("GRD08 failed Table 2 validation -- refusing to start the GUI.")
    # reuse the splash's hidden root so the splash survives until the UI is built
    root = _SPLASH_ROOT if _SPLASH_ROOT is not None else tk.Tk()
    setup_ui(root)
    app = ApatiteGUI(root)
    root.update_idletasks()                 # finish layout, THEN drop the splash
    if _SPLASH is not None:
        _SPLASH.destroy()
        _SPLASH = None
    if _SPLASH_ROOT is not None:
        root.deiconify()                    # main window was withheld until now
    # v2 A2: restore the last window geometry, clamped fully on-screen,
    # and save it back when the window closes
    try:
        m = re.match(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)",
                     str(load_ui_state().get("geometry", "")))
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            x, y = int(m.group(3)), int(m.group(4))
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            w, h = min(w, sw), min(h, sh - 40)      # leave the taskbar visible
            x = min(max(x, 0), max(sw - w, 0))
            y = min(max(y, 0), max(sh - h - 40, 0))
            root.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass

    def _on_close():
        try:
            save_ui_state(geometry=root.winfo_geometry())
        except Exception:
            pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", _on_close)
    # a .otproj passed on the command line (double-click / drag-onto-launcher /
    # file association) opens directly, skipping the startup dialog
    proj_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if proj_arg and proj_arg.lower().endswith(PROJ_EXT) and os.path.exists(proj_arg):
        root.after(100, lambda: app.open_project(proj_arg))
    else:
        root.after(100, app.show_startup_dialog)   # recent / browse / new blank
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()    # REQUIRED frozen (PyInstaller): the
    main()                              # multistart pool respawns the exe
