#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ot_figwin.py — reusable top-level matplotlib figure window for the OT P1 GUI
(spec #2/#3/#12, M2-1/M2-3, M3-1/M3-2).

FigureWindow = tk.Toplevel with an embedded matplotlib canvas plus:
  * navigation toolbar (interactive zoom/pan),
  * x/y min-max entries + Apply + Reset-to-auto (per axis or all axes),
  * Save button (SVG / PDF / 300-dpi PNG; vector first),
  * optional Clear hook (spec M2-4),
  * autosave(): timestamped vector copies into a project figures/ directory
    (never overwrites history — spec M3-2).

Windows are independent top-levels: several can be open side by side for
run-to-run comparison (spec M2-1) without blocking the main window.
"""
from __future__ import annotations

import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure


class FigureWindow(tk.Toplevel):
    def __init__(self, master, title, nrows=1, ncols=1, figsize=(7.5, 5.2),
                 dpi=96, figtype="figure", autosave_dir=None, on_clear=None,
                 extras=None):
        # extras: optional callable(bar_frame) — the caller adds its own
        # widgets (e.g. the multistart colormap selector) to the control bar
        super().__init__(master)
        self.title(title)
        self.figtype = figtype              # used in autosave filenames
        self.autosave_dir = autosave_dir
        self.on_clear = on_clear
        self.saved_paths = []               # everything this window wrote

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=2)
        self.fig = Figure(figsize=figsize, dpi=dpi)
        axes = self.fig.subplots(nrows, ncols)
        try:
            self.axes = list(axes.ravel())
        except AttributeError:              # single Axes
            self.axes = [axes]
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        tb.update()
        tb.pack(fill="x")

        # ---- axis-range controls (spec #1/#3, M2-3) ----
        self.ax_sel = tk.StringVar(value="all")
        if len(self.axes) > 1:
            ttk.Label(bar, text="axis").pack(side="left")
            ttk.Combobox(bar, textvariable=self.ax_sel, width=4,
                         state="readonly",
                         values=["all"] + [str(i + 1) for i in
                                           range(len(self.axes))]
                         ).pack(side="left", padx=(2, 6))
        self.rng = {}
        for lab in ("x min", "x max", "y min", "y max"):
            ttk.Label(bar, text=lab).pack(side="left")
            v = tk.StringVar(value="")
            ttk.Entry(bar, textvariable=v, width=8).pack(side="left",
                                                         padx=(2, 6))
            self.rng[lab] = v
        ttk.Button(bar, text="Apply", width=6,
                   command=self.apply_range).pack(side="left")
        ttk.Button(bar, text="Reset", width=6,
                   command=self.reset_range).pack(side="left", padx=(2, 8))
        if on_clear is not None:
            ttk.Button(bar, text="Clear", width=6,
                       command=self._clear).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Save...", width=8,
                   command=self.save_dialog).pack(side="right")
        if extras is not None:
            extras(bar)

    # ------------- axis ranges -------------
    def _selected_axes(self):
        s = self.ax_sel.get()
        if s == "all":
            return self.axes
        return [self.axes[int(s) - 1]]

    def apply_range(self):
        def val(key):
            t = self.rng[key].get().strip()
            return float(t) if t else None
        try:
            x0, x1 = val("x min"), val("x max")
            y0, y1 = val("y min"), val("y max")
        except ValueError:
            messagebox.showerror("Axis range", "ranges must be numbers",
                                 parent=self)
            return
        for ax in self._selected_axes():
            if x0 is not None or x1 is not None:
                ax.set_xlim(left=x0, right=x1)
            if y0 is not None or y1 is not None:
                ax.set_ylim(bottom=y0, top=y1)
        self.canvas.draw_idle()

    def reset_range(self):
        for ax in self._selected_axes():
            ax.relim()
            ax.autoscale()
        for v in self.rng.values():
            v.set("")
        self.canvas.draw_idle()

    def _clear(self):
        if self.on_clear:
            self.on_clear(self)
        self.canvas.draw_idle()

    # ------------- saving -------------
    def save_dialog(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Save figure", defaultextension=".svg",
            initialdir=os.environ.get("OT_OUT", os.getcwd()),
            initialfile=f"{self.figtype}.svg",
            filetypes=[("SVG vector", "*.svg"), ("PDF vector", "*.pdf"),
                       ("PNG 300 dpi", "*.png")])
        if not path:
            return
        try:
            self._savefig(path)
        except Exception as e:
            # a silent Tk-callback traceback reads as "nothing happened" in
            # the windowed exe — always surface the path + reason instead
            messagebox.showerror("Save figure", f"{path}\n\n{e}", parent=self)
            return
        self.saved_paths.append(path)
        messagebox.showinfo("Save figure", f"saved:\n{path}", parent=self)

    def _savefig(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        kw = {"dpi": 300} if path.lower().endswith(".png") else {}
        self.fig.savefig(path, **kw)

    def autosave(self, formats=("svg", "pdf")):
        """Timestamped copies into the project figures dir (spec M3-2).
        Filenames carry figtype + timestamp -> history is never overwritten."""
        if not self.autosave_dir:
            return []
        os.makedirs(self.autosave_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = []
        for ext in formats:
            p = os.path.join(self.autosave_dir, f"{self.figtype}_{stamp}.{ext}")
            self._savefig(p)
            out.append(p)
        self.saved_paths += out
        return out
