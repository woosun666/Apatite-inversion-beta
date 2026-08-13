# `apatite_gui.py` architecture

Architecture snapshot (2026-07-03, pre-UI-refactor); the v2 layout delta
below describes the shipped UI.

> **v2 layout delta (2026-08-03):**
> `setup_ui(root)` owns icon/fonts/optional sv_ttk theme (main + drivers +
> tests all call it); named fonts via `ui_font()` (the old
> `("TkDefaultFont", 9, "bold")` tuples resolved to a WRONG family);
> module-level matplotlib rcParams style all figures. Left panel =
> scrollable controls ABOVE + pinned readout/details block BELOW (outside
> the canvas); global bottom statusbar carries `self.status` + project
> mirror. Multistart tab = BOUNDS/Run (two column-pairs) → actions strip →
> results row (headline + `ms_table` Treeview + notes | project runs |
> envelope) → bundle figure with nav toolbar; `ms_text` is GONE —
> `_fill_ms_table()` reconfigures the Treeview per display (multistart vs
> envelope). Window geometry persists to `~/.ot_apatite_gui_ui.json`
> (clamped on-screen). Per-phase regression gate: `validate_gui_state.py`.

## Stack

- **GUI framework:** Tkinter + ttk (stdlib). No PyQt.
- **Plotting:** Matplotlib `TkAgg` backend, `FigureCanvasTkAgg`, four `Figure`s
  (Eye-fit 1x2, Melt volatiles 1x1, Viscosity preview 1x1, Multistart results 1x3).
- **Engine (never modified by the GUI):** `apatite_model.py` (forward physics),
  `apatite_inversion.py` (vectorised fixed-point forward `inv.forward_ratios`,
  multistart workers `inv._worker_init`/`inv._fit_one`, objective
  `inv.rmse_to_target` = RMSE to the hand-set target curve over XCl/XOH + XF/XOH),
  `giordano2008.py` (GRD08, startup self-check).

## Layout

One monolithic class `ApatiteGUI` plus module-level pure helpers:

- **Pure helpers (no Tk, unit-testable):** `compute_state` (one forward run →
  dict F/T_K/cloh/foh/fcl/conv/H2O/Cl/Fwt/sat_i), `load_tf_curve`,
  `load_observed` (flexible CSV headers; `is_zir` from a "Rock type"-like column
  containing "ZIR"), `viscosity_preview`, `results_to/from_csv_text`.
- **Left panel** (`_build_left_panel`): Project frame (Open/Save/Save As, run
  count label) → Data frame (observed CSV, T-F curve, const-T entry) →
  Forward parameters frame (9 params + K_spec; each row =
  label | − | ttk.Scale | + | entry, nudge step = slider range/20) →
  readout `tk.Text` ("-- current forward --" block, `_update_readout`) →
  status label (forward ms + convergence warning).
- **Recompute path:** every widget change → `_schedule_recompute()` (120 ms
  `root.after` debounce) → `_recompute()` → `compute_state` on the live grid
  (`_grid(NUM_F_LIVE=120)`: loaded T-F curve in "path" mode, Huber-2010
  analytic curve `am.generate_tf_curve` in "formula" mode, else const-T) →
  `_draw_eyefit` / `_draw_volatiles` / `_draw_viscosity` / `_update_readout`.
  Forward ≈ 145 ms.
- **T-F formula mode (2026-07-04):** third temperature mode next to
  path/const. `am.generate_tf_curve(T_initial, T_final, b)` evaluates
  T(F) = T_final + (T_initial−T_final)·F^(1/b) (inverse of Huber et al. 2010
  F = ((T−T_final)/(T_initial−T_final))^b; b=1/b=2 reproduce the digitized
  1000-750-b1/-B2 CSVs exactly). GUI wrapper `formula_tf_curve` returns the
  same `(F, T_K, T_of_F)` tuple as `load_tf_curve`; parameters live in a
  collapsible section of the Data frame (mini Canvas preview, Export CSV in
  Var1/Var2 format), persist in the manifest under the optional `tf_formula`
  key (version stays 1) and in each run record for exact replay.
- **Eye-fit axes convention (project-wide):** y = XCl/XOH on both panels;
  x = XF/XOH (left), XF/XCl (right). Matrix apatite grey dots, ZIR purple
  diamonds, target curve blue, H2O-sat onset orange star.
- **Multistart tab:** BOUNDS table → background thread `_ms_worker` fanning
  L-BFGS-B starts over a `ProcessPoolExecutor` (reuses engine worker code path,
  deterministic in `inv.SEED`) → queue-polled progress (`_ms_poll`) →
  `_show_ms_results` (param/rail table, family curve bundle, RMSE histogram).
  Objective = RMSE to the *current hand-set target curve* (Lormand stage 2),
  NOT to the observed scatter.
- **Projects (`.otproj`):** zip = `manifest.json` (version 1: params, k_spec,
  const_T, bounds, run settings, obs/tf paths + embedded copies, run index) +
  `data/` (embedded CSV copies) + `runs/run_NNN.csv` (Multioutput format).
  `open_project` restores everything and replays the latest run; new manifest
  keys are read with `.get(...)` defaults, so additions stay backward compatible.
- **Run history:** `self.proj["runs"]` list of dicts (time, note, params,
  k_spec, const_T, bounds, settings, best_rmse, results); Listbox
  `runs_list` + `_load_run` full replay; autosave after each completed run.

## UI-refactor ground rules

- Forward-model math must not change — spot-check: `compute_state` with
  `DEFAULT_PARAMS`/`DEFAULT_KSPEC`, const T = 800 °C, 120-pt grid gives
  cloh[0]=1.0602279855661303, cloh[-1]=0.034717894509228804,
  foh[0]=53.48308372977994, foh[-1]=4.267492450822285, sat_i=86.
- Multistart objective stays `inv.rmse_to_target` (engine); the ZIR misfit
  indicator is a *display* metric with a different definition —
  documented discrepancy, not merged.
