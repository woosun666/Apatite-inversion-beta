#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apatite_m1.py — OT P1 module M1: apatite EPMA data processing + review plots
(spec §5, §7.1-§7.3). Tk panel embedded as a tab of apatite_gui.py.

Science notes (spec §7.1):
  * X_F / X_Cl / X_OH are MONOVALENT-ANION-SITE mole fractions, sum = 1.
  * Recalc scheme = the SNCC protocol: **ApThermo (Weiran Li; Li & Costa 2020)
    26-oxygen-basis stoichiometry**, extracted cell-by-cell from the ApThermo
    reference workbook (2026-07-05) and verified during development against
    its Excel-computed golden rows (max|Δ| < 1e-5):
      - moles of oxygen per oxide (Ce2O3 x3, P2O5 x5, SiO2 x2, SO3 x3; F and
        Cl enter as wt/at.mass/2, i.e. 2 halogens ≡ 1 oxygen),
      - oxygen factor = 26 / Σ(mole O)   [Ca10(PO4)6X2 basis],
      - F_apfu = F/19 x factor, Cl_apfu = Cl/35.45 x factor,
      - OH_apfu = 2 − (F_apfu + Cl_apfu)  (anion site = 2 per formula),
      - X_i = apfu_i / 2;  calc H2O wt% = X_OH/X_F · (F wt%/19) · 18/2.
    Atomic masses are the ApThermo sheet's own values (F=19, Cl=35.45, ...) —
    do NOT "improve" them, point-identity with SNCC is the requirement.
  * OH-by-difference accumulates every other anion-column error — surfaced as
    a permanent note in the panel (spec M1-4).

Two ternaries (spec §7.2) use DELIBERATELY SEPARATE normalisations:
  * tern_fcls_wt  : F-Cl-S composition ternary (wt%, S from SO3; S sits on the
    tetrahedral site, NOT the halogen column — composition triangle only);
  * tern_anion    : X_F-X_Cl-X_OH mole fractions of the ONE anion column.
Do not merge these two functions (spec: 不可复用同一归一化函数).
"""
from __future__ import annotations

import csv
import io
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from ot_figwin import FigureWindow

# ---- constants --------------------------------------------------------------
# ApThermo (Li & Costa 2020) atomic/oxide masses — the SHEET'S OWN values;
# keep verbatim for point-identity with the SNCC dataset (spec §7.1).
APT_MASS = dict(CaO=56.08, Na2O=61.98, MgO=40.31, Ce2O3=328.24, MnO=70.94,
                FeO=71.85, SrO=103.62, P2O5=141.94, SiO2=60.08, SO3=80.06,
                F=19.0, Cl=35.45)
# moles of OXYGEN contributed per mole-of-oxide-mass unit (ApThermo cols X-AG)
APT_OXY = dict(CaO=1, Na2O=1, MgO=1, Ce2O3=3, MnO=1, FeO=1, SrO=1,
               P2O5=5, SiO2=2, SO3=3)
OXYGEN_BASIS = 26.0                     # Ca10(PO4)6X2
S_FROM_SO3 = 32.065 / 80.062            # wt% S = SO3 * this (display only)

RECALC_PROVISIONAL = False              # ApThermo-verified 2026-07-05
RECALC_NOTE = ("recalc = ApThermo (Li & Costa 2020) 26-oxygen stoichiometry, "
               "verified vs ApThermo reference-workbook golden rows. "
               "OH-by-difference accumulates all other anion errors.")

SUITES = ["", "early intrusive suite (pre-Fubilan)",
          "inter-mineralization intrusions", "Fubilan suite"]     # spec §7.3
ALTERATION_SAMPLES = {"24OT02-43", "24OT02-01", "24OT02-10"}      # spec §7.3

# target fields -> default source column (matches the OT EPMA export,
# sheet "Sheet1 (2)" of OT磷灰石EPMA结果-防灾-25.6.30.xlsx).
# REQUIRED drive the anion-site math; OPTIONAL oxides refine the oxygen sum
# (ApThermo includes them; missing -> 0, mapping may be left blank).
FIELDS = ["id", "F", "Cl", "SO3", "P2O5", "SiO2", "CaO", "Total"]
OPT_FIELDS = ["Na2O", "MgO", "Ce2O3", "MnO", "FeO", "SrO"]
DEFAULT_MAP = {"id": "Comment", "F": "F", "Cl": "Cl", "SO3": "SO3",
               "P2O5": "P2O5", "SiO2": "SiO2", "CaO": "CaO", "Total": "Total",
               "Na2O": "Na2O", "MgO": "MgO", "Ce2O3": "", "MnO": "MnO",
               "FeO": "FeO", "SrO": "SrO"}

WONG = {"blue": "#0072B2", "orange": "#E69F00", "sky": "#56B4E9",
        "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7",
        "yellow": "#F0E442", "black": "#000000"}
SUITE_COLOR = {SUITES[1]: WONG["sky"], SUITES[2]: WONG["orange"],
               SUITES[3]: WONG["verm"], "": WONG["blue"]}


# ============================ pure computation ================================
def load_epma(path, sheet=None):
    """Raw EPMA table -> DataFrame (all columns as read; numeric where possible)."""
    if path.lower().endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def recalc_anion_site(raw, mapping, total_lo=94.0, total_hi=102.0,
                      sio2_max=1.0):
    """Mapped raw table -> calc table with apfu, X_F/X_Cl/X_OH, derived ratios
    and quality flags (spec M1-2..M1-4, §7.1). Scheme = ApThermo (Li & Costa
    2020) 26-oxygen stoichiometry, cell-for-cell — see module header."""
    d = pd.DataFrame()
    d["id"] = raw[mapping["id"]].astype(str).str.strip()
    for f in FIELDS[1:]:
        d[f] = pd.to_numeric(raw[mapping[f]], errors="coerce")
    for f in OPT_FIELDS:                        # optional oxides: missing -> 0
        src = mapping.get(f, "")
        d[f] = (pd.to_numeric(raw[src], errors="coerce").fillna(0.0)
                if src in raw.columns else 0.0)
    d = d[d["id"].ne("") & d["P2O5"].notna()].reset_index(drop=True)
    d["sample"] = d["id"].str.rsplit("-", n=1).str[0]

    # --- ApThermo columns X..AJ: moles of oxygen per 100 g ---
    mole_O = sum(d[ox] / APT_MASS[ox] * APT_OXY[ox]
                 for ox in APT_OXY)             # oxides
    mole_O = mole_O + d["F"] / APT_MASS["F"] / 2.0 \
                    + d["Cl"] / APT_MASS["Cl"] / 2.0   # 2 halogens = 1 oxygen
    fac = OXYGEN_BASIS / mole_O                 # AM: oxygen factor (26 O)
    # --- anion site (AZ/BA/BB): 2 sites per Ca10(PO4)6X2 formula ---
    d["F_apfu"] = d["F"] / APT_MASS["F"] * fac
    d["Cl_apfu"] = d["Cl"] / APT_MASS["Cl"] * fac
    d["OH_apfu"] = 2.0 - (d["F_apfu"] + d["Cl_apfu"])   # OH by difference
    d["S_apfu"] = d["SO3"] / APT_MASS["SO3"] * fac      # P-site S (AX)
    d["S_wt"] = d["SO3"] * S_FROM_SO3
    d["X_F"] = d["F_apfu"] / 2.0
    d["X_Cl"] = d["Cl_apfu"] / 2.0
    d["X_OH"] = d["OH_apfu"] / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # ApThermo V: stoichiometric H2O wt% back-estimate
        d["calc_H2O_wt"] = (d["X_OH"] / d["X_F"] * d["F"]
                            / APT_MASS["F"] * 18.0 / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["F_Cl_wt"] = d["F"] / d["Cl"]
        d["XCl_XOH"] = d["X_Cl"] / d["X_OH"]
        d["XF_XOH"] = d["X_F"] / d["X_OH"]
        d["XF_XCl"] = d["X_F"] / d["X_Cl"]

    flags = []
    for _, r in d.iterrows():
        f = []
        if r["X_F"] + r["X_Cl"] > 1.0 or r["X_OH"] < 0.0:
            f.append("site>1")                   # over-filled anion column
        if not (total_lo <= r["Total"] <= total_hi):
            f.append("total")
        if r["SiO2"] > sio2_max:
            f.append("SiO2?zrn")                 # suspected zircon-host signal
        flags.append(",".join(f))
    d["flag"] = flags
    return d


def default_groups(calc):
    """Auto group defaults: ZHAI from id containing 'ZIR'; alteration from the
    fixed sample list; suite left empty (user assigns per spec §7.3 names)."""
    g = {}
    for _, r in calc.iterrows():
        g[r["id"]] = dict(
            zhai=("zir" in r["id"].lower()),
            suite="",
            alt=(r["sample"] in ALTERATION_SAMPLES))
    return g


# ---- ternary transforms: TWO separate normalisations (spec §7.2) ------------
def tern_fcls_wt(d, mF=1.0, mCl=1.0, mS=1.0):
    """F-Cl-S COMPOSITION ternary from wt% (S recomputed from SO3). Own
    normalisation — S is on the tetrahedral site, this is not a single-site
    mole-fraction triangle. Optional per-apex multipliers (annotated by caller)."""
    a = d["F"].to_numpy() * mF
    b = d["Cl"].to_numpy() * mCl
    c = d["S_wt"].to_numpy() * mS
    t = a + b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        fa, fb = a / t, b / t
    return _bary_xy(fa, fb)


def tern_anion(d):
    """X_F-X_Cl-X_OH ternary: anion-COLUMN mole fractions, already sum to 1 by
    construction (OH by difference). Distinct from tern_fcls_wt on purpose."""
    return _bary_xy(d["X_F"].to_numpy(), d["X_Cl"].to_numpy())


def _bary_xy(fa, fb):
    """Barycentric (A top, B bottom-left, C bottom-right) -> cartesian."""
    x = 0.5 * (2.0 * (1.0 - fa - fb) + fa)      # C right + half A
    y = np.sqrt(3.0) / 2.0 * fa
    return x, y


def draw_ternary_frame(ax, labels):
    v = np.array([[0.5, np.sqrt(3) / 2], [0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    ax.plot(v[:, 0], v[:, 1], "k-", lw=1)
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, labels[0], ha="center", fontsize=9)
    ax.text(-0.02, -0.03, labels[1], ha="right", fontsize=9)
    ax.text(1.02, -0.03, labels[2], ha="left", fontsize=9)
    for f in (0.2, 0.4, 0.6, 0.8):               # light grid
        ax.plot([f * 0.5, 1 - f * 0.5], [f * np.sqrt(3) / 2] * 2,
                color="0.85", lw=0.5, zorder=0)
        ax.plot([f, f + (1 - f) * 0.5], [0, (1 - f) * np.sqrt(3) / 2],
                color="0.85", lw=0.5, zorder=0)
        ax.plot([1 - f, 1 - f - (1 - f) * 0.5], [0, (1 - f) * np.sqrt(3) / 2],
                color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")


# ================================ Tk panel ====================================
class M1Panel(ttk.Frame):
    """The 'Apatite data' tab. app = the main App (for project dir/status)."""

    NUMERIC_PLOT_COLS = ["F", "Cl", "SO3", "S_wt", "P2O5", "SiO2", "CaO",
                         "Na2O", "MgO", "MnO", "FeO", "SrO",
                         "Total", "F_apfu", "Cl_apfu", "OH_apfu", "S_apfu",
                         "X_F", "X_Cl", "X_OH", "calc_H2O_wt", "F_Cl_wt",
                         "XCl_XOH", "XF_XOH", "XF_XCl"]

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.raw = None                  # imported DataFrame (as read)
        self.calc = None                 # recalculated table
        self.groups = {}                 # id -> {zhai, suite, alt}
        self.mapping = dict(DEFAULT_MAP)
        self.source = None
        self.total_lo = tk.DoubleVar(value=94.0)
        self.total_hi = tk.DoubleVar(value=102.0)
        self.sio2_max = tk.DoubleVar(value=1.0)
        self._windows = []
        self._build()

    # ---------------- UI ----------------
    def _build(self):
        # v2 C2: the three crowded control rows get labelled visual groups
        # (import/thresholds | grouping | plot windows) with separators
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="Import EPMA (CSV/XLSX)...",
                   command=self._import).pack(side="left")
        ttk.Button(top, text="Recalculate",
                   command=self._recalc).pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y",
                                                   padx=8, pady=3)
        for lab, var in (("Total in", self.total_lo), ("-", self.total_hi),
                         ("SiO2 >", self.sio2_max)):
            ttk.Label(top, text=lab).pack(side="left", padx=(6, 1))
            ttk.Entry(top, textvariable=var, width=6).pack(side="left")
        ttk.Button(top, text="Export calc CSV",
                   command=self._export_calc).pack(side="right")
        warn = ("⚠ " + RECALC_NOTE) if RECALC_PROVISIONAL else ""
        ttk.Label(self, text=warn, foreground="#a05000",
                  wraplength=900, justify="left").pack(fill="x", pady=(3, 3))

        # group editing controls
        gbar = ttk.Frame(self)
        gbar.pack(fill="x", pady=(0, 3))
        ttk.Label(gbar, text="selected rows →").pack(side="left")
        ttk.Button(gbar, text="ZHAI", width=6,
                   command=lambda: self._set_group(zhai=True)).pack(side="left")
        ttk.Button(gbar, text="matrix", width=7,
                   command=lambda: self._set_group(zhai=False)).pack(side="left")
        ttk.Separator(gbar, orient="vertical").pack(side="left", fill="y",
                                                    padx=8, pady=3)
        self.suite_var = tk.StringVar(value=SUITES[0])
        ttk.Combobox(gbar, textvariable=self.suite_var, values=SUITES,
                     width=32, state="readonly").pack(side="left", padx=(0, 4))
        ttk.Button(gbar, text="set suite",
                   command=lambda: self._set_group(suite=self.suite_var.get())
                   ).pack(side="left")
        ttk.Separator(gbar, orient="vertical").pack(side="left", fill="y",
                                                    padx=8, pady=3)
        ttk.Button(gbar, text="toggle alteration",
                   command=lambda: self._set_group(alt="toggle")
                   ).pack(side="left")

        # plots launcher row (three window groups, spec §5.3)
        pbar = ttk.Frame(self)
        pbar.pack(fill="x", pady=(0, 3))
        ttk.Label(pbar, text="plots:").pack(side="left", padx=(0, 4))
        ttk.Button(pbar, text="Ternary window (F-Cl-S + X_F-X_Cl-X_OH)",
                   command=self.win_ternary).pack(side="left")
        ttk.Button(pbar, text="Volatile-ratio window",
                   command=self.win_volatile).pack(side="left", padx=4)
        ttk.Button(pbar, text="F/Cl vs element / custom window",
                   command=self.win_custom).pack(side="left")
        ttk.Separator(pbar, orient="vertical").pack(side="left", fill="y",
                                                    padx=8, pady=3)
        ttk.Label(pbar, text="custom x/y:").pack(side="left", padx=(0, 2))
        self.cx = tk.StringVar(value="XF_XOH")
        self.cy = tk.StringVar(value="XCl_XOH")
        for v in (self.cx, self.cy):
            ttk.Combobox(pbar, textvariable=v, values=self.NUMERIC_PLOT_COLS,
                         width=9, state="readonly").pack(side="left", padx=1)
        self.clogx = tk.BooleanVar(value=False)
        self.clogy = tk.BooleanVar(value=False)
        ttk.Checkbutton(pbar, text="log x", variable=self.clogx).pack(side="left")
        ttk.Checkbutton(pbar, text="log y", variable=self.clogy).pack(side="left")

        # table (v2 C2: content-sized columns — suite/flag were huge and
        # empty — numeric columns right-aligned, zebra striping)
        cols = ("id", "zhai", "suite", "alt", "X_F", "X_Cl", "X_OH", "flag")
        self.tv = ttk.Treeview(self, columns=cols, show="headings", height=18)
        widths = (130, 56, 150, 40, 66, 70, 66, 96)
        for c, w in zip(cols, widths):
            self.tv.heading(c, text=c)
            self.tv.column(c, width=w,
                           anchor=("e" if c.startswith("X_") else "w"),
                           stretch=(c in ("suite", "flag")))
        self.tv.tag_configure("odd", background="#f2f2f5")
        vs = ttk.Scrollbar(self, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=vs.set)
        self.tv.pack(side="left", fill="both", expand=True)
        vs.pack(side="left", fill="y")

    # ---------------- data ----------------
    def _import(self):
        path = filedialog.askopenfilename(
            title="EPMA table", filetypes=[("EPMA table", "*.csv *.xlsx *.xlsm")])
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xlsm")):
                xl = pd.ExcelFile(path)
                sheet = xl.sheet_names[0]
                if len(xl.sheet_names) > 1:
                    sheet = _pick_sheet(self, xl.sheet_names)
                    if sheet is None:
                        return
                self.raw = load_epma(path, sheet)
            else:
                self.raw = load_epma(path)
        except Exception as e:
            messagebox.showerror("Import", str(e))
            return
        self.source = os.path.basename(path)
        if not _map_dialog(self, self.raw.columns.tolist(), self.mapping):
            return
        self._recalc(fresh_groups=True)

    def _recalc(self, fresh_groups=False):
        if self.raw is None:
            messagebox.showinfo("M1", "import an EPMA table first")
            return
        missing = [f for f in FIELDS if self.mapping.get(f)
                   not in self.raw.columns]
        if missing:
            messagebox.showerror("M1", "unmapped/missing columns: "
                                 + ", ".join(missing))
            return
        try:
            self.calc = recalc_anion_site(
                self.raw, self.mapping, float(self.total_lo.get()),
                float(self.total_hi.get()), float(self.sio2_max.get()))
        except Exception as e:
            messagebox.showerror("Recalc", str(e))
            return
        if fresh_groups or not self.groups:
            self.groups = default_groups(self.calc)
        else:                            # keep edits, add defaults for new ids
            for k, v in default_groups(self.calc).items():
                self.groups.setdefault(k, v)
        self._refresh_table()
        nfl = int((self.calc["flag"] != "").sum())
        self.app.status.config(
            text=f"M1: {len(self.calc)} analyses, {nfl} flagged ({self.source})")

    def _refresh_table(self):
        self.tv.delete(*self.tv.get_children())
        for n, (_, r) in enumerate(self.calc.iterrows()):
            g = self.groups.get(r["id"], {})
            self.tv.insert("", "end", iid=r["id"], values=(
                r["id"], "ZHAI" if g.get("zhai") else "matrix",
                g.get("suite", ""), "alt" if g.get("alt") else "",
                f"{r['X_F']:.3f}", f"{r['X_Cl']:.4f}", f"{r['X_OH']:.3f}",
                r["flag"]), tags=("odd",) if n % 2 else ())

    def _set_group(self, zhai=None, suite=None, alt=None):
        ids = self.tv.selection()
        if not ids:
            messagebox.showinfo("Groups", "select rows in the table first")
            return
        for i in ids:
            g = self.groups.setdefault(i, dict(zhai=False, suite="", alt=False))
            if zhai is not None:
                g["zhai"] = zhai
            if suite is not None:
                g["suite"] = suite
            if alt == "toggle":
                g["alt"] = not g["alt"]
        self._refresh_table()

    def _export_calc(self):
        if self.calc is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="apatite_calc.csv")
        if path:
            self.calc_with_groups().to_csv(path, index=False)
            self.app.status.config(text=f"written {os.path.basename(path)}")

    def calc_with_groups(self):
        d = self.calc.copy()
        d["zhai"] = [self.groups.get(i, {}).get("zhai", False) for i in d["id"]]
        d["suite"] = [self.groups.get(i, {}).get("suite", "") for i in d["id"]]
        d["alteration"] = [self.groups.get(i, {}).get("alt", False)
                           for i in d["id"]]
        return d

    # ---------------- project persistence ----------------
    def get_state(self):
        return dict(mapping=self.mapping, source=self.source,
                    total_lo=float(self.total_lo.get()),
                    total_hi=float(self.total_hi.get()),
                    sio2_max=float(self.sio2_max.get()), groups=self.groups)

    def set_state(self, st, raw_csv_text=None):
        self.mapping = st.get("mapping", dict(DEFAULT_MAP))
        self.source = st.get("source")
        self.total_lo.set(st.get("total_lo", 94.0))
        self.total_hi.set(st.get("total_hi", 102.0))
        self.sio2_max.set(st.get("sio2_max", 1.0))
        self.groups = st.get("groups", {})
        if raw_csv_text:
            self.raw = pd.read_csv(io.StringIO(raw_csv_text))
            self._recalc()

    def raw_csv_text(self):
        return self.raw.to_csv(index=False) if self.raw is not None else None

    # ---------------- plotting ----------------
    def _scatter(self, ax, x, y, d):
        """Group/flag-aware scatter (spec M1-8, §7.3): suite = colour, ZHAI =
        triangle vs matrix circle, alteration = open symbol, flagged = grey x."""
        zh = np.array([self.groups.get(i, {}).get("zhai", False)
                       for i in d["id"]])
        alt = np.array([self.groups.get(i, {}).get("alt", False)
                        for i in d["id"]])
        su = np.array([self.groups.get(i, {}).get("suite", "")
                       for i in d["id"]])
        fl = (d["flag"] != "").to_numpy()
        for suite in np.unique(su):
            col = SUITE_COLOR.get(suite, WONG["blue"])
            for is_z, mk in ((False, "o"), (True, "^")):
                for is_a in (False, True):
                    m = (su == suite) & (zh == is_z) & (alt == is_a) & ~fl
                    if not m.any():
                        continue
                    lab = f"{suite or 'unassigned'} {'ZHAI' if is_z else 'matrix'}"
                    ax.scatter(np.asarray(x)[m], np.asarray(y)[m], s=26,
                               marker=mk,
                               facecolor="none" if is_a else col,
                               edgecolor=col, linewidth=0.8,
                               label=lab + (" (alt)" if is_a else ""))
        if fl.any():
            ax.scatter(np.asarray(x)[fl], np.asarray(y)[fl], s=18, marker="x",
                       color="0.6", linewidth=0.8, label="quality-flagged")

    def _need_calc(self):
        if self.calc is None:
            messagebox.showinfo("M1", "import + recalculate first")
            return True
        return False

    def _new_win(self, title, figtype, nrows=1, ncols=1, figsize=(7.5, 5.2)):
        w = FigureWindow(self.app.root, title, nrows, ncols, figsize=figsize,
                         figtype=figtype, autosave_dir=self.app.figures_dir())
        self._windows.append(w)
        return w

    def win_ternary(self):
        if self._need_calc():
            return
        w = self._new_win("M1 ternaries", "m1_ternary", 1, 2, (10.5, 5.2))
        axA, axB = w.axes
        draw_ternary_frame(axA, ("F (wt%)", "Cl (wt%)", "S (wt%)"))
        x, y = tern_fcls_wt(self.calc)
        self._scatter(axA, x, y, self.calc)
        axA.set_title("F–Cl–S composition ternary (wt%; S on tetrahedral site)",
                      fontsize=8)
        draw_ternary_frame(axB, ("X$_F$", "X$_{Cl}$", "X$_{OH}$"))
        x, y = tern_anion(self.calc)
        self._scatter(axB, x, y, self.calc)
        axB.set_title("anion-site mole fractions (sum = 1; OH by difference)",
                      fontsize=8)
        axA.legend(fontsize=5, loc="upper left", bbox_to_anchor=(-0.05, 1.0))
        w.fig.tight_layout()
        w.canvas.draw_idle()
        w.autosave()

    def win_volatile(self):
        if self._need_calc():
            return
        w = self._new_win("M1 volatile ratios", "m1_volatile", 1, 2, (10.5, 5))
        d = self.calc
        for ax, xcol, xlab in ((w.axes[0], "XF_XOH", "X$_F$/X$_{OH}$"),
                               (w.axes[1], "XF_XCl", "X$_F$/X$_{Cl}$")):
            self._scatter(ax, d[xcol], d["XCl_XOH"], d)
            ax.set_xlabel(xlab)
            ax.set_ylabel("X$_{Cl}$/X$_{OH}$")
            ax.grid(alpha=0.3)
        w.axes[0].legend(fontsize=5)
        w.fig.tight_layout()
        w.canvas.draw_idle()
        w.autosave()

    def win_custom(self):
        if self._need_calc():
            return
        cx, cy = self.cx.get(), self.cy.get()
        w = self._new_win(f"M1 custom {cy} vs {cx}", "m1_custom", 1, 1)
        ax = w.axes[0]
        d = self.calc
        self._scatter(ax, d[cx], d[cy], d)
        if self.clogx.get():
            ax.set_xscale("log")
        if self.clogy.get():
            ax.set_yscale("log")
        ax.set_xlabel(cx)
        ax.set_ylabel(cy)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5)
        w.fig.tight_layout()
        w.canvas.draw_idle()
        w.autosave()


# ---------------- small dialogs ----------------
def _pick_sheet(parent, names):
    top = tk.Toplevel(parent)
    top.title("Pick sheet")
    var = tk.StringVar(value=names[0])
    ttk.Label(top, text="Excel sheet:").pack(padx=10, pady=(10, 2))
    ttk.Combobox(top, textvariable=var, values=names,
                 state="readonly", width=24).pack(padx=10)
    out = {}

    def ok():
        out["s"] = var.get()
        top.destroy()
    ttk.Button(top, text="OK", command=ok).pack(pady=8)
    top.grab_set()
    parent.wait_window(top)
    return out.get("s")


def _map_dialog(parent, columns, mapping):
    """Column-mapping dialog (spec M1-1). Required fields must map; optional
    ApThermo oxides may stay blank (-> 0 wt%). Returns True on OK."""
    top = tk.Toplevel(parent)
    top.title("Column mapping")
    vars_ = {}
    for r, f in enumerate(FIELDS + OPT_FIELDS):
        opt = f in OPT_FIELDS
        ttk.Label(top, text=f + (" (opt)" if opt else ""), width=12).grid(
            row=r, column=0, padx=6, pady=2, sticky="w")
        v = tk.StringVar(value=mapping.get(f, "")
                         if mapping.get(f) in columns else "")
        ttk.Combobox(top, textvariable=v, values=([""] if opt else []) + columns,
                     width=22).grid(row=r, column=1, padx=6)
        vars_[f] = v
    ok = {"v": False}

    def accept():
        miss = [f for f in FIELDS if vars_[f].get() not in columns]
        if miss:
            messagebox.showerror("Mapping", "unmapped: " + ", ".join(miss),
                                 parent=top)
            return
        for f, v in vars_.items():
            mapping[f] = v.get()
        ok["v"] = True
        top.destroy()
    ttk.Button(top, text="OK", command=accept).grid(
        row=len(FIELDS), column=0, columnspan=2, pady=8)
    top.grab_set()
    parent.wait_window(top)
    return ok["v"]
