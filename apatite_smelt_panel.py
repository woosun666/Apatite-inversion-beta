#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apatite_smelt_panel.py — the "Melt S" tab of apatite_gui.py.

Thin Tk wrapper around `apatite_smelt.py`: it runs that module's pipeline in
a worker thread and displays the result (figures A/B/C/D, the unit-level
statistics table, the run log). ALL science lives in `apatite_smelt.py` —
this file computes nothing, so the GUI can never disagree with the CLI.

The panel lives in its own module (not inside apatite_smelt.py) so the
analysis module stays free of tkinter and usable headless.

Two contracts worth knowing when editing:
- `apatite_smelt` aborts on bad data via `sys.exit`, which is correct for a
  CLI but would kill the GUI process. `_worker` catches SystemExit and turns
  it into a message.
- Progress output goes through `apatite_smelt.log()`; the worker swaps in a
  sink that queues lines for the Tk thread, then restores the previous sink.

The dFMQ firewall applies to everything shown here: dFMQ is an INPUT of the
conversion, melt S is never evidence of oxidation state, no S valence is
asserted, and the numbers are concentrations only (flux/tonnage = P3).
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import apatite_smelt as asm
from ot_figwin import FigureWindow

# short radio labels — the full titles live in asm.FIG_TITLES and on the axes
FIG_LABELS = {"A": "A check", "B": "B main", "C": "C vs dFMQ",
              "D": "D vs XF/XCl"}


class SmeltPanel(ttk.Frame):
    """The 'Melt S' tab. app = the main App (status bar, figures dir)."""

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.res = None                  # last run_analysis() result dict
        self.last_run = None             # summary kept in the .otproj
        self.fig_key = tk.StringVar(value="B")
        self.out_dir = tk.StringVar(value=asm.OUT_DIR)
        self.reprep = tk.BooleanVar(value=False)
        self.do_golden = tk.BooleanVar(value=False)
        self.sources = {}                # {tag: path} raw-input overrides
                                         # (empty = canonical defaults)
        self._q = queue.Queue()
        self._busy = False
        self._windows = []
        self._build()

    # ---------------- UI ----------------
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.btn_run = ttk.Button(top, text="Run melt-S calculation",
                                  command=self._run)
        self.btn_run.pack(side="left")
        ttk.Checkbutton(top, text="rebuild inputs",
                        variable=self.reprep).pack(side="left", padx=(8, 2))
        ttk.Checkbutton(top, text="golden check",
                        variable=self.do_golden).pack(side="left", padx=2)
        ttk.Button(top, text="Data sources...",
                   command=self._pick_sources).pack(side="left", padx=(8, 2))
        self.lbl_src = ttk.Label(top, text="", foreground="#a05000")
        self.lbl_src.pack(side="left", padx=(0, 2))
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y",
                                                   padx=8, pady=3)
        ttk.Label(top, text="figure").pack(side="left", padx=(0, 3))
        for key in asm.FIG_KEYS:
            ttk.Radiobutton(top, text=FIG_LABELS[key], value=key,
                            variable=self.fig_key,
                            command=self._draw).pack(side="left")
        ttk.Button(top, text="Pop out",
                   command=self._popout).pack(side="right")
        ttk.Button(top, text="Save figures",
                   command=self._save_figs).pack(side="right", padx=4)

        out = ttk.Frame(self)
        out.pack(fill="x", pady=(3, 0))
        ttk.Label(out, text="output dir").pack(side="left")
        ttk.Entry(out, textvariable=self.out_dir).pack(side="left", fill="x",
                                                       expand=True, padx=4)
        ttk.Button(out, text="...", width=3,
                   command=self._pick_out).pack(side="left")
        ttk.Button(out, text="Open folder", width=12,
                   command=self._open_out).pack(side="left", padx=(4, 0))

        guard = ttk.Label(self, text="⚠ " + asm.GUARD_SENTENCE,
                          foreground="#a05000", justify="left")
        guard.pack(fill="x", pady=(4, 2))
        # wrap to the real width: a fixed wraplength clips on narrow windows
        guard.bind("<Configure>",
                   lambda e: e.widget.configure(wraplength=max(200,
                                                               e.width - 12)))

        pane = ttk.PanedWindow(self, orient="vertical")
        pane.pack(fill="both", expand=True)

        figwrap = ttk.Frame(pane)
        bar = ttk.Frame(figwrap)
        bar.pack(fill="x")
        self.lbl_run = ttk.Label(bar, text="not run yet", foreground="#555555")
        self.lbl_run.pack(side="left", padx=(2, 0))
        self.fig = Figure(figsize=(8, 4.4), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, figwrap)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        if hasattr(self.app, "_attach_nav"):
            self.app._attach_nav(self.canvas, bar)
        pane.add(figwrap, weight=3)

        bottom = ttk.Frame(pane)
        cols = ("unit", "type", "n", "median", "P16", "P84", "censored",
                "inherited")
        widths = (200, 60, 40, 80, 80, 80, 70, 70)
        tabf = ttk.Frame(bottom)
        tabf.pack(side="left", fill="both", expand=True)
        self.tv = ttk.Treeview(tabf, columns=cols, show="headings", height=8)
        for c, w in zip(cols, widths):
            self.tv.heading(c, text=c)
            self.tv.column(c, width=w, anchor=("w" if c in ("unit", "type")
                                               else "e"),
                           stretch=(c == "unit"))
        self.tv.tag_configure("odd", background="#f2f2f5")
        self.tv.tag_configure("empty", foreground="#999999")
        vs = ttk.Scrollbar(tabf, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=vs.set)
        self.tv.pack(side="left", fill="both", expand=True)
        vs.pack(side="left", fill="y")

        logf = ttk.LabelFrame(bottom, text="run log", padding=2)
        logf.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.txt = tk.Text(logf, width=52, height=8, font=("Consolas", 8),
                           state="disabled", wrap="none")
        ls = ttk.Scrollbar(logf, orient="vertical", command=self.txt.yview)
        hs = ttk.Scrollbar(logf, orient="horizontal", command=self.txt.xview)
        self.txt.configure(yscrollcommand=ls.set, xscrollcommand=hs.set)
        hs.pack(side="bottom", fill="x")   # log lines carry long paths
        self.txt.pack(side="left", fill="both", expand=True)
        ls.pack(side="left", fill="y")
        pane.add(bottom, weight=2)

        self._empty_axes("press Run to compute melt S")

    # ---------------- helpers ----------------
    def _empty_axes(self, msg):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, msg, transform=self.ax.transAxes,
                     ha="center", va="center", fontsize=10, color="#777777")
        self.canvas.draw_idle()

    def _status(self, msg, color="#006400"):
        if getattr(self.app, "status", None) is not None:
            self.app.status.config(text=msg, foreground=color)

    def _append_log(self, line):
        self.txt.configure(state="normal")
        self.txt.insert("end", line + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _pick_out(self):
        d = filedialog.askdirectory(title="melt-S output directory",
                                    initialdir=self.out_dir.get())
        if d:
            self.out_dir.set(d)

    def _open_out(self):
        d = self.out_dir.get()
        if not os.path.isdir(d):
            messagebox.showinfo("Melt S", "no output directory yet -- run "
                                          "the calculation first")
            return
        os.startfile(d)                  # Windows-only app (per README_EXE)

    # ---------------- data sources (user 2026-08-08) ----------------
    SRC_SPEC = (
        ("zircon", "zircon dFMQ/T (grouped CSV)",
         (("CSV files", "*.csv"), ("All files", "*.*"))),
        ("zhai", "ZHAI inclusions (Excel / normalized CSV)",
         (("Excel or normalized CSV", "*.xlsx *.xlsm *.csv"),
          ("All files", "*.*"))),
        ("matrix", "matrix/phenocryst EPMA (Excel / normalized CSV)",
         (("Excel or normalized CSV", "*.xlsx *.xlsm *.csv"),
          ("All files", "*.*"))),
    )

    def _default_src(self, tag):
        return {"zircon": asm.ZIRCON_CSV, "zhai": asm.ZHAI_XLSX,
                "matrix": asm.MATRIX_XLSX}[tag]

    def _effective_src(self, tag):
        return self.sources.get(tag) or self._default_src(tag)

    def _refresh_src_label(self):
        custom = sorted(self.sources)
        self.lbl_src.config(text=("custom sources: " + ", ".join(custom))
                            if custom else "")

    def _pick_sources(self):
        """Data-source dialog: one row per raw input — editable path,
        Browse..., a live exists-indicator; Reset restores the canonical
        defaults (env / script-side resolution). OK keeps only picks that differ
        from the defaults; the engine rebuilds the normalized input caches
        automatically whenever the recorded cache source differs from the
        requested file (run_prep provenance guard)."""
        dlg = tk.Toplevel(self)
        dlg.title("Melt S data sources")
        dlg.transient(self.winfo_toplevel())
        frm = ttk.Frame(dlg, padding=8)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Inputs of the melt-S pipeline. Defaults = the "
                            "canonical dataset (OT_SMELT_* env override, "
                            "else files next to the script). A source may be either the "
                            "RAW file (zircon grouped CSV / ZHAI xlsx / "
                            "matrix xlsx) or an already-normalized "
                            "smelt_input_*.csv table (auto-detected and "
                            "ingested as-is, prep skipped). Changed sources "
                            "rebuild the input caches automatically; the "
                            "golden check only makes sense on the canonical "
                            "dataset.",
                  wraplength=560, justify="left").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        vars_, marks = {}, {}

        def check(tag):
            ok = os.path.exists(vars_[tag].get().strip())
            marks[tag].config(text="✓" if ok else "missing",
                              foreground="#006400" if ok else "#a00000")

        for i, (tag, label, ftypes) in enumerate(self.SRC_SPEC, start=1):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w",
                                            padx=(0, 6), pady=2)
            v = tk.StringVar(value=self._effective_src(tag))
            vars_[tag] = v
            ttk.Entry(frm, textvariable=v, width=68).grid(row=i, column=1,
                                                          sticky="we", pady=2)
            marks[tag] = ttk.Label(frm, width=8)
            marks[tag].grid(row=i, column=2, padx=4)

            def browse(t=tag, ft=ftypes):
                cur = vars_[t].get().strip()
                p = filedialog.askopenfilename(
                    parent=dlg, title=f"choose the {t} source file",
                    initialdir=(os.path.dirname(cur) if cur else None),
                    filetypes=list(ft))
                if p:
                    vars_[t].set(p)

            ttk.Button(frm, text="Browse...", width=9,
                       command=browse).grid(row=i, column=3, pady=2)
            v.trace_add("write", lambda *a, t=tag: check(t))
            check(tag)
        frm.columnconfigure(1, weight=1)

        def reset():
            for tag, _label, _ft in self.SRC_SPEC:
                vars_[tag].set(self._default_src(tag))

        def ok():
            picked = {}
            for tag, _label, _ft in self.SRC_SPEC:
                p = vars_[tag].get().strip()
                if p and not asm._same_path(p, self._default_src(tag)):
                    picked[tag] = p
            self.sources = picked
            self._refresh_src_label()
            self._append_log(
                "[sources] " + (", ".join(f"{t} <- {p}" for t, p
                                          in sorted(picked.items()))
                                if picked else "canonical defaults"))
            dlg.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=len(self.SRC_SPEC) + 1, column=0, columnspan=4,
                  sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Reset to canonical defaults",
                   command=reset).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="OK", width=8, command=ok).pack(side="left")
        ttk.Button(btns, text="Cancel", width=8,
                   command=dlg.destroy).pack(side="left", padx=(4, 0))
        # scripted-drive hooks (cf. FigureWindow w._style): the file dialogs
        # are modal and undriveable, the closures are not
        dlg._src_vars, dlg._src_ok, dlg._src_reset = vars_, ok, reset
        dlg.grab_set()
        return dlg

    # ---------------- run ----------------
    def _run(self):
        if self._busy:
            return
        self._busy = True
        self.btn_run.config(state="disabled")
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")
        if asm.in_temp_dir(self.out_dir.get()):
            # a doomed dir, e.g. a _MEI path saved by a pre-26.8.8 exe run:
            # anything written there is invisible at best, deleted on app
            # exit at worst -- redirect to the safe default, loudly
            self.out_dir.set(asm.OUT_DIR)
            self._append_log("[out] output dir was inside the OS temp dir "
                             "-- redirected to " + asm.OUT_DIR)
        if self.sources and self.do_golden.get():
            self._append_log("[golden] WARNING: custom data sources are "
                            "active -- the golden reference is keyed to the "
                            "canonical dataset, mismatches are expected")
        self._status("melt S: running...", "#a05a00")
        threading.Thread(target=self._worker,
                         args=(self.out_dir.get(), self.reprep.get(),
                               self.do_golden.get(), dict(self.sources)),
                         daemon=True).start()
        self.after(60, self._poll)

    def _worker(self, out_dir, reprep, golden, sources):
        prev = asm.set_log_sink(lambda m: self._q.put(("log", m)))
        try:
            if not asm.validate():
                raise RuntimeError("spot checks failed -- refusing to run")
            asm._selfcheck_alteration()
            res = asm.run_analysis(out_dir, reprep=reprep,
                                   sources=sources or None)
            if golden:
                asm.golden_compare(res["points"])
            self._q.put(("done", res))
        except SystemExit as e:           # the module's data guards abort
            self._q.put(("fail", str(e) or "aborted by a data guard"))
        except Exception as e:
            self._q.put(("log", traceback.format_exc()))
            self._q.put(("fail", f"{type(e).__name__}: {e}"))
        finally:
            asm.set_log_sink(prev)

    def _poll(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._finish(payload)
                    return
                elif kind == "fail":
                    self._busy = False
                    self.btn_run.config(state="normal")
                    self._status(f"melt S failed: {payload}", "#a00000")
                    self._empty_axes("run failed -- see the log")
                    messagebox.showerror("Melt S", payload)
                    return
        except queue.Empty:
            pass
        self.after(60, self._poll)

    def _finish(self, res):
        self._busy = False
        self.btn_run.config(state="normal")
        self.res = res
        pts = res["points"]
        n_smelt = sum(1 for p in pts if not p["censored"])
        self.last_run = dict(
            when=datetime.now().isoformat(timespec="seconds"),
            out_dir=res["out_dir"], n_points=len(pts), n_smelt=n_smelt,
            unit_stats=res["unit_stats"], sources=dict(self.sources))
        self._fill_table(res["unit_stats"])
        self._draw()
        self.lbl_run.config(
            text=f"{len(pts)} points ({n_smelt} with melt S) -- "
                 f"{self.last_run['when']}")
        self._status(f"melt S: {n_smelt} points computed -> "
                     f"{os.path.basename(res['out_dir'])}")

    # ---------------- display ----------------
    def _fill_table(self, ustats, restored=False):
        self.tv.delete(*self.tv.get_children())
        for i, r in enumerate(ustats):
            tags = ["odd"] if i % 2 else []
            if not r["n"]:
                tags.append("empty")
            vals = (asm.UNIT_DISPLAY.get(r["unit"], r["unit"]), r["type"],
                    r["n"],
                    "" if r["smelt_median"] is None
                    else f"{r['smelt_median']:.1f}",
                    "" if r["smelt_p16"] is None else f"{r['smelt_p16']:.1f}",
                    "" if r["smelt_p84"] is None else f"{r['smelt_p84']:.1f}",
                    r["n_censored"], r["n_inherited_host_excluded"])
            self.tv.insert("", "end", values=vals, tags=tuple(tags))
        if restored:
            self._append_log("[project] unit statistics restored from the "
                             "project file; press Run to redraw the figures")

    def _draw(self):
        if self.res is None:
            return
        self.ax.clear()
        self.ax.set_axis_on()
        asm.draw_figure(self.fig_key.get(), self.ax, self.res["points"],
                        self.res["sample_k"], footnote=False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _popout(self):
        if self.res is None:
            messagebox.showinfo("Melt S", "run the calculation first")
            return
        key = self.fig_key.get()
        figures_dir = (self.app.figures_dir()
                       if hasattr(self.app, "figures_dir") else None)
        w = FigureWindow(self.app.root, f"Melt S -- {asm.FIG_TITLES[key]}",
                         1, 1, figsize=asm.FIG_SIZES[key],
                         figtype=f"smelt_fig{key}", autosave_dir=figures_dir)
        asm.draw_figure(key, w.axes[0], self.res["points"],
                        self.res["sample_k"])
        w.fig.tight_layout()
        w.canvas.draw_idle()
        self._windows.append(w)

    def _save_figs(self):
        if self.res is None:
            messagebox.showinfo("Melt S", "run the calculation first")
            return
        prev = asm.set_log_sink(self._append_log)
        try:
            asm.write_figures(self.res["out_dir"], self.res["points"],
                              self.res["sample_k"])
        except Exception as e:
            messagebox.showerror("Melt S", str(e))
        finally:
            asm.set_log_sink(prev)
        self._status(f"melt S figures written -> {self.res['out_dir']}")

    # ---------------- project persistence ----------------
    def get_state(self):
        """Summary only: melt S is fully reproducible from the canonical
        input files, so the project stores what was run, not the points."""
        if self.last_run is None and not self.sources:
            return None
        return dict(fig=self.fig_key.get(), out_dir=self.out_dir.get(),
                    sources=dict(self.sources), last_run=self.last_run)

    def set_state(self, st):
        if not st:
            return
        self.fig_key.set(st.get("fig", "B"))
        self.sources = dict(st.get("sources") or {})
        self._refresh_src_label()
        if self.sources:
            self._append_log("[sources] restored from the project: "
                             + ", ".join(f"{t} <- {p}" for t, p
                                         in sorted(self.sources.items())))
        if st.get("out_dir"):
            d = st["out_dir"]
            if asm.in_temp_dir(d):       # _MEI path from a pre-26.8.8 exe
                d = asm.OUT_DIR
            self.out_dir.set(d)
        lr = st.get("last_run")
        if lr:
            self.last_run = lr
            if lr.get("unit_stats"):
                self._fill_table(lr["unit_stats"], restored=True)
            self.lbl_run.config(
                text=f"restored: {lr.get('n_points', '?')} points, "
                     f"run {lr.get('when', '?')} (press Run to redraw)")
