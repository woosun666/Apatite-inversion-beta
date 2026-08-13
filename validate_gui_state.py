#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_gui_state.py — regression gate for GUI-only refactors.

Guards the UI-refactor ground rule (ARCHITECTURE.md): UI work must not touch
the forward model. Checks, in order:

  1. byte-compile apatite_gui / apatite_m1 / ot_figwin
  2. compute_state golden numbers (the ARCHITECTURE.md ground-rule block):
     DEFAULT_PARAMS / DEFAULT_KSPEC, const T = 800 C, 120-pt grid ->
     cloh[0], cloh[-1], foh[0], foh[-1], sat_i must match to 1e-12 relative
  3. widget smoke test (needs a display; skip with --no-gui): ApatiteGUI
     builds, the debounced recompute produces a state, every notebook tab
     selects, the window destroys cleanly

Exit 0 = all pass.  Run after EVERY UI phase before committing.
"""
import argparse
import os
import py_compile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GOLD = dict(cloh0=1.0602279855661303, clohN=0.034717894509228804,
            foh0=53.48308372977994, fohN=4.267492450822285, sat_i=86)
RTOL = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true",
                    help="skip the Tk widget smoke test (headless use)")
    args = ap.parse_args()

    for m in ("apatite_gui.py", "apatite_m1.py", "ot_figwin.py",
              "apatite_smelt.py", "apatite_smelt_panel.py"):
        py_compile.compile(os.path.join(HERE, m), doraise=True)
    print("[1/3] byte-compile OK")

    import numpy as np
    import apatite_gui as ag

    F = np.linspace(1.0, 0.1, 120)
    T_K = np.full(120, 800.0 + 273.15)
    s = ag.compute_state(ag.DEFAULT_PARAMS, ag.DEFAULT_KSPEC, F, T_K)
    got = dict(cloh0=s["cloh"][0], clohN=s["cloh"][-1],
               foh0=s["foh"][0], fohN=s["foh"][-1], sat_i=s["sat_i"])
    for k, want in GOLD.items():
        g = got[k]
        if k == "sat_i":
            ok = g == want
        else:
            ok = abs(g - want) <= RTOL * abs(want)
        if not ok:
            sys.exit(f"[2/3] GOLDEN MISMATCH {k}: got {g!r}, want {want!r}")
    print("[2/3] compute_state golden OK")

    if args.no_gui:
        print("[3/3] widget smoke test SKIPPED (--no-gui)")
        return
    import tkinter as tk
    root = tk.Tk()
    root.geometry("+0+0")
    ag.setup_ui(root)                 # fonts/theme path used by a real launch
    app = ag.ApatiteGUI(root)
    t0 = time.time()
    while app.state is None and time.time() - t0 < 10:
        root.update()                 # let the 120 ms debounce fire
        time.sleep(0.02)
    assert app.state is not None, "recompute never produced a state"
    for tab in app.nb.tabs():
        app.nb.select(tab)
        root.update()
    tabs = [app.nb.tab(t, "text") for t in app.nb.tabs()]
    for want in ("Apatite data (M1)", "Melt S"):
        assert want in tabs, f"tab missing: {want} (have {tabs})"
    root.destroy()
    print(f"[3/3] widget smoke test OK (state + {len(tabs)} tabs)")


if __name__ == "__main__":
    main()
