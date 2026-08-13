# Apatite Cl-OH-F explorer — user manual (`apatite_gui.py`)

Interactive GUI for the Lormand apatite volatile model, following the workflow of
the original MATLAB `Apatite_V1_*.m` scripts. The MATLAB original has no widget
GUI — its interface is *edit parameters in the script → run an operation → look
at the figures → iterate*. This GUI turns that loop into live sliders and plots,
while **reusing the validated Python engine unchanged** (`apatite_model.py`
forward physics, `apatite_inversion.py` multistart, `giordano2008.py` GRD08).

---

## 1. Provenance and citation

The underlying model reproduces (Python port, validated against the MATLAB
golden output to max|rel| ≈ 8×10⁻⁵):

> Lormand, C., Humphreys, M.C.S., Coumans, J.P., Colby, D.J., Chelle-Michou, C.
> & Li, W. — *Volatile budgets and evolution in porphyry-related magma systems,
> determined using apatite.* Lithos (2024).

built on:

> Humphreys, M.C.S., et al. (2021) — *Rapid pre-eruptive mush reorganisation and
> atmospheric volatile emissions from the 12.9 ka Laacher See eruption,
> determined using apatite.* EPSL 576, 117198.

The original model was developed in the ERC STEMMS project (Horizon 2020 grant
No 864923, PI M. Humphreys). **Cite the originals; check licensing before any
public release of this port.**

Viscosity preview: Giordano, Russell & Dingwell (2008), EPSL 271, 123-134
(coefficients verified against the paper's Tables 1-2; `validate_grd08.py`).

---

## 2. What the model computes

For a cooling/crystallising melt (melt fraction F from 1 → 0.1):

1. **Volatile evolution** — H₂O, Cl, F in the melt follow Rayleigh
   fractionation (crystal-melt D's). When dissolved H₂O reaches the saturation
   value `H2O_Sat`, a fluid exsolves; Cl and F partition into it (fluid-melt
   D's) and the fluid salinity (NaCl-eq) is tracked.
2. **Melt speciation** — total H₂O is split into molecular H₂O + OH using the
   regressed speciation constant `K_spec`.
3. **Apatite partitioning** — the apatite X-site mole fractions (X_Cl, X_OH,
   X_F) are solved at every F step from the melt mole fractions via the
   **non-ideal** Cl-OH-F exchange model (Margules W parameters + ΔG(T) of the
   Cl-OH and F-OH exchanges — hence temperature matters).
4. The model outputs the apatite ratio curves **XCl/XOH, XF/XOH, XF/XCl vs F**,
   which are compared with measured apatite compositions.

**Two-stage Lormand inversion (unchanged here):**
*Stage 1* — hand-tune the forward parameters until the model curve passes
through the observed apatite ratio cloud (**eye fit**; the observations are
overplotted for the eye and are NOT in any automatic objective).
*Stage 2* — a multistart L-BFGS-B fit **to that hand-set target curve** within
parameter BOUNDS, quantifying parameter non-uniqueness (trade-offs) around the
target. See §8 for what the resulting "family" does and does not mean.

---

## 3. Requirements & launch

* Python ≥ 3.9 with numpy, scipy, pandas, matplotlib (already required by the
  repo) — tkinter ships with Python. No extra installs.
* Launch — three ways:

```
cd D:\code\oktedi_magma_viscosity
python apatite_gui.py                # terminal
python apatite_gui.py my.otproj      # open a project directly
```

or simply **double-click `apatite_gui.bat`** in the repo folder (uses
`pythonw` when available, so no console window). For one-click access,
right-click the .bat → *Send to → Desktop (create shortcut)*. You can also
**drag a `.otproj` file onto the .bat** (or associate the `.otproj` extension
with it) to open that project directly, skipping the startup dialog.

On startup a dialog offers: **recent projects** (name / folder / age columns;
double-click or Enter to open, Escape = blank session) / **Browse...** /
**Open sample project** (generates and loads the synthetic demo data) /
**New blank session**. See §7.1 for project files.

Look-and-feel (UI spec v2, 2026-08-03): the GUI applies the Sun Valley ttk
theme when the optional `sv-ttk` package is installed (`pip install sv-ttk`;
without it the stock theme is used). The window position/size persists per
machine in `~/.ot_apatite_gui_ui.json`. A bottom status bar shows the project
name and the forward/status messages; every figure tab carries a compact
zoom/pan toolbar (top right).

Optional environment variables (preload paths at startup):

| var            | meaning                                            |
|----------------|----------------------------------------------------|
| `OT_OBS`       | observed apatite ratio CSV                         |
| `OT_TF`        | T-F cooling-curve CSV                              |
| `OT_OUT`       | base directory for save dialogs and exports        |
| `OT_SMELT_OUT` | Melt S output directory (overrides the default)    |

Default export/output locations never resolve into the OS temp directory:
in the single-file exe the unpack dir (`%TEMP%\_MEI…`) is **deleted when
the app exits**, so anything written there is lost (2026-08-08 fix). When
neither `OT_OUT` nor a usable working directory is available, exports fall
back to `%USERPROFILE%\apa-inversion-beta-output\`.

The GUI refuses to start if the GRD08 Table-2 self-check fails.

---

## 4. Input file formats

### 4.1 Observed apatite ratios (CSV)

Flexible headers; either style works (case-insensitive):

* OT style (`test_OT.csv`): columns `F/OH`, `Cl/OH`, `F/Cl` + `Rock type`
  (rows whose Rock type contains `ZIR` are plotted as zircon-hosted; blank
  Rock type rows are dropped).
* Generic style: `Cl_OH`, `F_OH`, `F_Cl` (+ optional `group`).

Ratios are **X-site mole ratios** — the EPMA wt% → mole-ratio recalculation is
done by the user beforehand (deliberately not part of this code).

### 4.2 T-F cooling curve (CSV)

Same format as the MATLAB `TF_File`: two columns headed `Var1` (T in °C) and
`Var2` (melt fraction F). Handling follows the engine/MATLAB rules: duplicate F
rows deduplicated (first kept), F range = max(F) → max(0.1, min(F)), T linearly
interpolated onto the model grid. Without a curve, the model runs isothermally
at the **const T (degC)** entry (equivalent to the MATLAB hardwired `T = ...`).
Temperature matters: the apatite exchange ΔG is T-dependent, so a real cooling
path bends the ratio curves relative to an isothermal run.

Instead of loading a CSV you can also **generate** the cooling path from the
Huber et al. (2010) parameterization — see §4.2b.

### 4.2b T-F formula (Huber et al. 2010)

The third temperature mode generates the T-F curve from the generalized
melt-fraction function of Huber et al. (2010):

    F = ((T − T_final) / (T_initial − T_final)) ^ b

where `F` is the melt fraction (0–1), `T_initial` / `T_final` are the
*effective* starting and ending temperatures of the modeled crystallization
interval, and `b` is the cooling coefficient (`b=1` linear; `b=2` = the "B2"
curve of the OT 26.6.25 run — the previously used digitized
`1000-750-b1.csv` / `...-B2.csv` files are reproduced exactly by `b=1` /
`b=2`). The curve is evaluated analytically on the F grid via
`T(F) = T_final + (T_initial − T_final)·F^(1/b)`.

**What the temperature bounds mean (revised per review):**

* `T_initial` (~1000 °C in the OT runs) is **not** the liquidus of a hydrous
  granitic melt. It is the onset of fractional crystallization of a
  mafic-to-intermediate parental magma at ~15 km depth, which progressively
  differentiates toward the volatile-enriched felsic melts of porphyry Cu
  reservoirs; ~1000 °C is well established for basaltic-to-andesitic arc
  magmas (Collins et al., 2021). The modeled volatile trajectories cover the
  full differentiation interval from this parental composition through
  apatite saturation and volatile exsolution.
* `T_final` is the *effective near-solidus end* of the modeled interval. For
  the PMD end-member it is ~790 °C, guided by Ti-in-zircon thermometry on
  ore-related porphyry magmas (Schiller & Finger, 2019) and recent
  experimental constraints on similar systems (Zhang et al., 2020; Xiao et
  al., 2024; Grocolas et al., 2025).

The parameters sit in the collapsible **T-F formula (Huber 2010)** section of
the Data panel (it unfolds automatically when the mode is selected). A mini
preview redraws on every change (press **Enter** or leave the entry).
**Export CSV** saves the generated curve in the `Var1`/`Var2` format of §4.2,
so it can be re-loaded as a file or fed to the command-line inversion
wrappers. The three parameters are saved in the project manifest and in every
run record, so runs replay exactly.

### 4.3 Parameter CSV (optional load)

`Load params CSV` accepts either a MATLAB-format `Multioutput_*.csv` row
(headers `H2O, Cl, F, Dxtl-mCl, …, H2OSat, RMSE`; row 1 = best fit) or an
engine-key file such as `OT_inv_bestfit_params.csv` (headers `H2O_init, …`;
`K_spec` is also read if present).

---

## 5. Forward parameters (left panel)

The 9 fitted parameters + speciation constant, exactly the MATLAB
`InputParam` block:

| parameter    | meaning                                                        | typical |
|--------------|----------------------------------------------------------------|---------|
| `H2O_init`   | initial melt H₂O, wt%                                          | 3–5     |
| `Cl_init`    | initial melt Cl, wt%                                           | 0.2–0.6 |
| `F_init`     | initial melt F, wt%                                            | 0.1–0.5 |
| `D_xl_m_Cl`  | crystal-melt distribution coefficient, Cl                      | 0.02–0.4|
| `D_xl_m_F`   | crystal-melt distribution coefficient, F (>1 = compatible)     | 0.5–2   |
| `D_xl_m_H2O` | crystal-melt distribution coefficient, H₂O                     | ~0.12   |
| `D_fl_m_Cl`  | fluid-melt distribution coefficient, Cl (drives the Cl "strip")| 15–30   |
| `D_fl_m_F`   | fluid-melt distribution coefficient, F                         | 1.5–5   |
| `H2O_Sat`    | H₂O saturation (solubility) value, wt% — sets the sat onset    | 8–11    |
| `K_spec`     | OH-H₂O speciation constant                                     | 0.36–0.47|

`D_fl_m_PrimBoilH2O` (primary boiling) is fixed at 0 (off), as in all OT runs.

The parameters sit in **four collapsible groups** (click a group header to
fold/unfold; the fold state is saved in the project): *Initial melt
composition*, *Crystal/melt partition coefficients*, *Fluid/melt partition
coefficients*, *Model parameters*. Each parameter has a slider (coarse, with
its min/max printed at the two ends), **− / + step buttons** (one click = 1/20
of the slider range; **Ctrl+click = 10×** the step) and an entry box (exact;
press **Enter**, or roll the **mouse wheel** over it for ±1 step). Hovering a
parameter name shows a tooltip with its meaning, typical range and reference
(edit the `PARAM_INFO` dict in `apatite_gui.py` to change the texts).

Plots refresh live (~ms; a 120 ms debounce coalesces rapid slider drags). The
structured readout — pinned at the bottom of the left panel, always visible
while the controls above it scroll (v2 A2) — lists T mode, melt start/end H₂O-Cl-F,
the saturation onset (F and, with a T-F curve, T), the Cl peak, the model and
observed Cl/OH + F/OH ranges, and the ZIR misfit. (The former "⚠ not covered"
range-coverage marker was removed 2026-08-04: on real OT data the model range
never brackets the full scatter by design.) The raw text log lives behind the
*details (raw log)* expander. A red `[FIXED-POINT NOT CONVERGED]` warning
appears if the partition solver failed for the current parameters — do not
trust the curves in that state.

**Temperature mode** is explicit (radio buttons in the Data panel): *T–F path
(from file)*, *constant T* or *T–F formula (Huber 2010)* (§4.2b). The inactive
inputs are greyed out; choosing *constant T* keeps a loaded T-F file
(deactivated, not deleted). Loading a T-F file switches to path mode
automatically; selecting the formula mode unfolds its parameter section. The
readout's first row shows which source is active (`T path` / `T formula` /
`const T`).

Buttons: **Reset** (back to the 26.6.25 working set) · **Load params CSV**
(§4.3) · **Export bundle** (§7).

---

## 6. The four tabs vs the three MATLAB operations

| MATLAB operation                                 | GUI                       |
|--------------------------------------------------|---------------------------|
| `Explore model and/or create multistart target`   | **Explore target curve** tab (+ Save) |
| `Run multistart to a target`                      | **Multistart** tab        |
| `Visualize multistart outputs`                    | **Multistart** tab (below)|
| *(not in MATLAB)*                                 | Melt volatiles, Viscosity, Melt S |

### 6.1 Explore target curve (stage 1; the "Eye-fit" tab before 2026-08-04)

Two panels, **linear axes** (project convention): y = XCl/XOH on both, x =
XF/XOH (left) and XF/XCl (right). Grey dots = matrix apatite, purple diamonds = zircon-hosted
(ZIR), blue line = current target curve, orange star = H₂O-saturation onset
(annotated with its F and, in path mode, T). A single shared legend sits at
the top of the figure. Iterate the sliders until the curve passes through the
cloud you believe (for OT: the ZIR/high-Cl/OH end = early melt). The MATLAB
equivalent is its figure 2, redrawn per script run — here it is live.

Extras:

* **misfit (ZIR, med. dist.)** in the left panel corner — the median
  range-normalized orthogonal distance from the ZIR points to the curve,
  recomputed on every forward run. NB this is a *display* metric; the
  multistart objective is still the RMSE to the hand-set target curve (§6.4).
* **log X** checkbox (top right) switches the XF/XCl panel to a log x-axis
  (useful when the data crowd into the low-XF/XCl corner); saved per project.
* **Click any observed point** to pop up its source CSV row (sample ID and
  raw values); click the popup (or wait 8 s) to dismiss it.
* Overlaid comparison curves from the run history (§7.1) and the multistart
  top-N (§6.4) are drawn dashed/semi-transparent; the active curve stays
  solid blue.
* **Plot / Clear buttons** (top left): **Clear** drops every overlay (run
  overlays, multistart top-N, envelope previews) and blanks both panels;
  **Plot** recomputes the forward model from the current sliders and redraws
  everything fresh. Useful after an envelope preview or many history overlays
  have cluttered the panels.

### 6.2 Melt volatiles

Melt H₂O (wt%) and Cl, F (×10) vs melt fraction F, with the saturation line —
the "what does the melt do" view of the same parameters (equivalent to
`plot_OT_melt_evolution.py`, live).

**Pop out** (2026-08-07) opens the same panel in an independent figure window
with the standard axis-range controls (x/y min–max + Apply/Reset, zoom/pan
toolbar) plus a line-style editor: pick a line (or *all*), set its width,
*colour…* opens a colour picker; the legend follows the restyle. The window
is a snapshot of the current parameters — reopen it after editing them.
Styling is per-window (the embedded tab keeps the default style); use the
window's *Save…* / autosave for the styled figure.

### 6.3 Viscosity preview

Feeds the current volatile trajectory into GRD08 as **two side-by-side
panels** (2026-08-08, mirroring the P1 melt-only main figure): **(a) log η
against melt fraction F** (1 → 0.1, crystallisation to the right) and
**(b) log η against temperature** (cooling to the right), sharing one y
axis (labels on (a) only) with a single legend on (a). Each panel shows the
coupled (enriching-H₂O) path and the saturation line (F_sat / T_sat). The
constant-H₂O comparison line was **removed** (2026-08-08); its role is
carried by the thin grey dashed **constant-H₂O reference isopleths
(3/4/5/6 wt%)** — same path, only H₂O frozen — wt%-labelled on the outer
panel only. With a T-F curve loaded (or the Huber formula selected), T
varies along the path and both panels draw; in **const T** mode η(T)
degenerates to a point, so panel (b) shows a hint instead (the title states
the mode). **Caveats printed in the title:** majors are the PLACEHOLDER
`ot_majors` (not an OT result), and dissolved H₂O above ~8 wt% is beyond
the GRD08 calibration (extrapolated). This is a quick-look, not the
production figure (`ot_viscosity_volatile_overlay.py`).

**Pop out** works as on the Melt volatiles tab (§6.2): independent snapshot
window (here 1×2, same layout) with axis-range controls and the line
width/colour editor; picking a line by name restyles it on **both** panels
at once.

### 6.4 Multistart (stage 2 + visualisation)

1. **BOUNDS table** — lb/ub per parameter (MATLAB `lb`/`ub`). The
   *center on current params (±30 %)* button brackets the current eye-fit;
   widen individual bounds as needed.
2. **starts / processes** — number of L-BFGS-B start points (MATLAB
   `numStart`; 100 for a quick look, 400+ for production) and worker processes
   (background process pool; the GUI stays responsive and shows per-start
   progress). Deterministic in the engine seed (`inv.SEED`): the same
   bounds/starts reproduce the same family, any worker count.
   **Run-time display**: at launch the status line shows an upfront estimate
   (`est ~2m 30s`, based on the per-start time measured on this machine in
   previous runs); while running it shows elapsed time and the remaining-time
   extrapolation (`running 120/400  1m 05s  ~2m 32s left`), self-correcting as
   throughput settles; on completion, the total wall time. The very first run
   on a machine shows `(estimating...)` until the first starts complete. NB
   with a process pool the first few seconds include worker startup, so the
   early ETA reads slightly high.
3. **Run multistart** — the objective is the RMSE to the *current* target
   curve over XCl/XOH + XF/XOH (XF/XCl excluded), exactly the MATLAB objective.
4. **Results** (MATLAB `MultistartFigures` equivalents; since v2 the tab is
   laid out as BOUNDS + Run settings on top, an actions strip, then a results
   row = table | project runs | envelope, with the bundle figure below.
   Results row and figure sit in a **vertical split pane** (2026-08-04):
   drag the sash to trade table space for figure space — the figure pane
   gets all extra height by default, and the panel descriptions sit in a
   caption line BELOW the figure instead of in-axes titles):
   * parameter table (a real table since v2 A1, 6 rows visible + its own
     scrollbar — the old monospace text block wrapped unreadably) — lb/ub,
     accepted-family min/median/max and a **rail flag** (>30 % of accepted
     fits within 2 % of a bound → widen that bound and re-run; rail rows are
     highlighted orange);
   * melt initial-volatile ranges of the accepted family
     (RMSE ≤ *accept ≤ x best*, default 5×) on the full-width notes line;
   * curve bundle over the observations, **Lormand-MATLAB style (readme
     Fig 7): ALL successful runs** (spread-sampled to ≤300 for speed),
     coloured by log10(RMSE) (colorbar; an orange dashed threshold on the
     bar marks the N-th best — everything on the better side is the top-N,
     2026-08-06), worst drawn first so the good
     curves sit on top; the observed points draw ABOVE the whole fan
     (2026-08-04 — they used to be buried under the curves). The **colormap
     combobox** and the **pt size** spinbox (observed marker size; ZIR draws
     1.5×) sit together on the actions strip and in the popout bundle bar
     (2026-08-04): colormap = jet (default, the Lormand-MATLAB look) /
     turbo / viridis / plasma / coolwarm / Greys; both restyle every open
     bundle in place, no re-run.
     The width of the fan is set by the high-RMSE runs, i.e. by how wide
     the BOUNDS are — the accepted family itself always hugs the target
     curve (the objective IS the target curve; honest data-wide spread
     still needs the multi-target envelope workflow) + RMSE histogram on a
     **log x-axis** with the cutoff dashed.
5. **Export Multioutput CSV** — MATLAB column format (`H2O, Cl, F, Dxtl-mCl,
   …, H2OSat, RMSE`, ascending RMSE); rows feed straight into
   `apatite_model.py --params` or the batch scripts.
6. **Apply best fit to sliders** — loads the lowest-RMSE parameter set back
   into the left panel and returns to the Explore-target-curve tab for another iteration.
7. **top-N overlay** (entry next to *starts*; default 5) — after a run (or a
   replay) the best N solutions are overlaid on the target-curve panels as
   semi-transparent orange curves, making solution non-uniqueness visible.
   *Clear overlays* (Project runs box) removes them.
8. **tolerance (ftol)** (default `1e-10` = engine-tight) — the L-BFGS-B
   convergence tolerance, i.e. the *Tolerance* field of the Lormand MATLAB
   GUI (its default is `1e-2`). With the tight default nearly every start
   converges fully onto the target curve, so the fan is thin; a loose value
   (`1e-2`) stops runs early and the fan spreads over the data like the
   MATLAB readme Fig 7. **Loose runs are DISPLAY/exploration only**: the
   best fit and the family statistics are degraded (a warning line appears
   in the results text) — never report a loose-tolerance family width, and
   re-run with the tight default before exporting numbers.
9. **explore draws** (default 0 = off) — N extra random parameter sets drawn
   uniformly in the BOUNDS and forwarded WITHOUT optimisation, underlaid in
   the bundle with the same log-RMSE colouring. This is the raw curve space
   the bounds allow (prior-predictive) — the widest, cheapest data-enveloping
   fan; also purely visual, not stored in the run history.

### 6.4b Objective metric, diagnostics window, bundle popout (P1 spec)

* **objective** combobox: `logrmse` (default; RMSE on log10-transformed
  ratios, P1 spec §7.4) or `rmse` (plain — ALL runs up to the 26.6.25
  canonical used this). ⚠ The two are NOT comparable; the active metric is
  shown in every results label and stored per run/project. Command-line
  scripts keep `rmse` unless passed `metric="logrmse"`.
* **Save figures** (2026-08-06): writes the three embedded panels (the two
  ratio fans + the metric histogram) as 300-dpi PNG + PDF — one combined
  file plus each panel separately — into a visible timestamped folder
  `exports/YYYYMMDD_HHMMSS/` under `OT_OUT` (or the cwd), same convention
  as the Explore-tab export bundle, then opens that folder in Explorer.
  Use this for on-disk copies; the popout *autosaves* below are packed
  inside the `.otproj` instead.
* **RMSE diagnostics window** (spec M2-5, Lormand-style): sorted runs +
  histogram with the acceptance cutoff, metric vs each of the 9 parameters
  (bounds as x-limits — rail patterns are obvious), and a Cl_init–F_init
  pair panel coloured by log10(metric). Metric = the run objective.
* **Pop out bundle window** (spec M2-1/M2-3): the multistart bundle in an
  independent window — target curve, all runs coloured by log10(metric)
  with the same colorbar as the tab, top-N bold and on top (since
  2026-08-06 its colour FOLLOWS the colormap — no separate colour; the
  orange dashed threshold on the colorbar is what singles the top-N out),
  observations. Several popouts can be compared side by
  side. All popout windows carry a navigation toolbar, x/y min–max + Apply +
  Reset-to-auto, Save (SVG/PDF/300-dpi PNG) and a timestamped autosave.
  ⚠ Autosaves go to a session temp folder and are packed INSIDE the
  `.otproj` zip under `figures/` when the project is saved — no visible
  `figures/` folder ever appears on disk (rename the `.otproj` to `.zip`
  and extract to retrieve them). For an on-disk file use the popout's
  **Save…** button or the tab's **Save figures**; the status bar states
  where every autosave went. The bundle popout bar additionally carries the
  **colormap** selector (shared with the embedded bundle — changing it
  recolours both, colorbars and the top-N legend swatch included).

### 6.6 Apatite data (M1) tab — EPMA processing + review plots

1. **Import EPMA (CSV/XLSX)** — sheet picker + column-mapping dialog
   (defaults match the OT EPMA export: `Comment/F/Cl/SO3/P2O5/SiO2/CaO/
   Total`; S arrives as SO3 wt%).
2. **Recalculate** — anion-site mole fractions X_F/X_Cl/X_OH via the
   **ApThermo (Li & Costa 2020) 26-oxygen stoichiometry**, extracted
   cell-for-cell from the ApThermo reference workbook and pinned during
   development by a golden regression against Excel-computed rows
   (machine-precision match). This is the SNCC-protocol recalculation used
   for the manuscript's apatite comparison.
   Optional oxides (Na2O/MgO/Ce2O3/MnO/FeO/SrO) refine the
   oxygen sum; also outputs apfu, S apfu (P site) and the stoichiometric
   calc-H2O wt%. Quality flags: `site>1` (X_F+X_Cl>1 or X_OH<0), `total`
   (outside the editable window), `SiO2?zrn` (suspected zircon-host
   contamination). OH-by-difference accumulates all other anion errors.
   Provenance NB: the historical `test_OT.csv` ratios
   agree with this recalc to ≤1% (F/OH) / ≤6% (Cl/OH) — consistent with
   rounded wt% inputs when that file was generated; regenerate from M1 if
   exact ratios are wanted.
3. **Groups** — select rows → ZHAI/matrix, suite (fixed names), toggle
   alteration (24OT02-43/-01/-10 pre-marked). NB the EPMA Comment column has
   no ZIR marker, so ZHAI must be assigned here (or arrives with a future
   groups-file import).
4. **Plot windows** (three independent groups): ternary pair
   (F–Cl–S wt% composition AND X_F–X_Cl–X_OH anion-site — two DIFFERENT
   normalisations), volatile-ratio pair, and F/Cl-vs-element /
   custom x-y (log axes). Symbols: suite = colour, ZHAI = triangle vs matrix
   circle, alteration = open symbol, flagged = grey ×. Every window
   autosaves SVG+PDF (packed inside the `.otproj` under `figures/` on
   project save — see the autosave note in §6.4b; use **Save…** for an
   on-disk copy).
5. Raw + calculated tables and all M1 state persist in the `.otproj`.

### 6.7 Melt S tab — apatite S → melt S (Tassara et al. 2026)

> **Read the orange banner on the tab.** ΔFMQ is an **input term** of the
> conversion, so melt S is never independent evidence of oxidation state; no
> apatite S valence (S⁶⁺/S²⁻) is asserted anywhere; the numbers are
> **concentrations** — flux/tonnage/endowment are out of scope (P3).

A thin front end for `apatite_smelt.py`: the tab computes nothing itself, so
its numbers cannot drift from the command line. Everything is read from the
canonical data files (zircon ΔFMQ/T grouped CSV, the ZHAI pairing workbook,
the ApThermo apatite sheet), not from what is loaded in the other tabs.
**Data sources...** (2026-08-08) opens a dialog to point any of the three
inputs at a different file — editable path + *Browse...* per row, a live
exists-indicator, and *Reset to canonical defaults* (the `OT_SMELT_*` env /
script-side resolution). Each source may be either the RAW file (zircon
grouped CSV / ZHAI workbook / matrix workbook) **or an already-normalized
`smelt_input_*.csv` table** (recognised by its header and ingested as-is,
prep skipped — e.g. an edited copy of a cache; its `unit` column is used
directly, and the Edinburgh/uniqueness guards still apply). Non-default picks show in orange next to the button,
are logged with every run, and are stored in the project file. The
normalized input caches rebuild automatically whenever a cache's recorded
`# source:` provenance differs from the requested file, in either
direction — switching datasets can never silently mix caches. The golden
check is keyed to the canonical dataset: with custom sources active it is
expected to fail, and the run log says so before the run starts.

1. **Run melt-S calculation** — runs in a worker thread; progress appears in
   the *run log* pane. **rebuild inputs** re-reads the three raw sources
   instead of the normalized cache; **golden check** additionally compares
   against the reference Excel calculator (`--golden`).
2. **Figure A / B / C / D** — the radio row switches the embedded figure.
   A = apatite S vs ΔFMQ behaviour check; **B = the main panel**, melt S by
   canonical unit (circles matrix, triangles ZHAI, grey open triangles =
   inherited-host ZHAI, shown but excluded from boxes and statistics);
   C = melt S vs ΔFMQ, descriptive only; **D = melt S vs apatite XF/XCl**
   (manuscript candidate, rulings 2026-08-06: x = the same analysis's X-site
   mole ratio, no shared input with the conversion; per-unit boxplots
   (whis 5–95%, n ≥ 3) at each group's median XF/XCl + one combined log-log
   fit line, slope 95% CI (cluster bootstrap by sample) in the legend —
   fit numbers stream into the run log; no bands (removed by ruling);
   inherited-host ZHAI and censored points omitted). **Pop out** opens the
   selected
   figure in a FigureWindow (axis controls + SVG/PDF/PNG save + autosave into
   the project `figures/`); **Save figures** writes all four as PDF+PNG into
   the output dir. Embedded figures omit the footnote for space — the popout
   and saved copies carry it, including the guard sentence.
3. **Unit statistics table** — n / median / P16 / P84 per unit per type, plus
   censored (SO₃ ≤ d.l., never substituted) and inherited-host counts. Grey
   rows are empty combinations (e.g. Sydney has no ZHAI analyses).
4. **Output dir** — CSVs (`OT_smelt_points/sample_k/unit_stats`) and the QC
   report land here; *Open folder* shows them. Defaults (override with
   `OT_SMELT_OUT`): script/onedir run → `apatite_smelt_out/` beside the
   app; **single-file exe** → `apatite_smelt_out/` under `OT_OUT`, or
   `%USERPROFILE%\apa-inversion-beta-output\apatite_smelt_out` when
   `OT_OUT` is unset — never the `%TEMP%\_MEI…` unpack dir, which is
   deleted on exit (2026-08-08 data-loss fix). A dir inside the OS temp
   tree (e.g. restored from a project saved by a pre-fix exe) is
   auto-redirected to the default, with a note in the run log.
5. The project stores *what was run* (settings + unit statistics + timestamp),
   not the point table: melt S is fully reproducible from the canonical
   inputs. Reopening shows the restored numbers; press Run to redraw figures.

Absolute melt S is good to ~2× (Tassara et al., 2026) — report the
**relative** contrasts between units, not the absolute ppm.

### 6.5 Envelope (26.6.24 multi-target honest uncertainty)

A single hand-set target gives an artificially narrow family (parameter
trade-off only — the objective fits the TARGET, not the scatter). The honest
melt-volatile range is the **union of the accepted families of three
eye-validated targets bracketing the observed cloud** (`ot_target_envelope.py`,
run OT-test-26.6.24_envelope). The *Envelope (3 targets)* box wires that
workflow into the GUI:

1. Tune the sliders so the curve threads the LOWER edge of the (matrix) cloud
   → **set low**. Repeat through the middle (**set central**) and the upper
   edge (**set high**). Or: set central only and click **suggest low/high
   from central** (scales F_init ×0.72/1.30, Cl_init ×0.80/1.30 — the 26.6.24
   factors; a starting point, retune by eye).
2. **preview on target tab** overlays the three forward curves (dashed,
   blue/black/orange) on the observations — the eye-validation step. Iterate
   1–2 until the band wraps the data.
3. Widen the BOUNDS so they contain all three targets (the run refuses
   otherwise — a clipped family would fake a narrow envelope), then
   **Run envelope (3× multistart)**: the three runs execute sequentially with
   live progress; results show per-target families, per-target log-RMSE
   histograms, and the **UNION melt H2O/Cl/F range — the number to report**.
4. CSVs are auto-exported next to the observed file
   (`gui_env_{low,central,high,union}.csv`); the three targets persist in the
   `.otproj`. Envelope runs are not added to the run history (single-target
   replays only).

---

## 7. Outputs

* **Export bundle** (left panel) → a timestamped folder
  `exports/YYYYMMDD_HHMMSS/` under `OT_OUT` (or the cwd) containing
  `target_curve.csv` (`Frac_Melt, XCl_div_XOH, XF_div_XOH, XF_div_XCl` — same
  as the engine target format), `params.csv` (the 9 parameters + K_spec), and
  the target-curve figure as `eyefit.png` (300 dpi) + `eyefit.pdf`
  (supplementary-material ready).
* **Export Multioutput CSV** (Multistart tab) → MATLAB-format multistart table.

### 7.1 Project files (`.otproj`)

A project bundles the whole working session into **one portable file** (a zip):

* `manifest.json` — current parameters, K_spec, const-T, BOUNDS, run settings,
  and the run index;
* `data/` — **embedded copies** of the observed-ratio CSV and T-F curve, so the
  project opens on any machine even if the original files moved (the original
  paths are also recorded and preferred when they still exist);
* `runs/run_NNN.csv` — the complete multistart family of every run, in
  Multioutput format.

The Melt S tab adds an optional `smelt` block to `manifest.json` (settings +
last-run summary). Projects saved before it existed open unchanged, and older
builds ignore the extra key.

Workflow:

* **Project panel** (top left): `Open…` / `Save` / `Save As…`. The label shows
  the current project and its run count.
* Every completed multistart is **appended to the project run history** with a
  timestamp and the optional **run note** (entry next to the Run button) — the
  project accumulates the target-iteration log. With a project file set, each
  run **autosaves** immediately.
* **Project runs** list (Multistart tab): each line shows the run number,
  timestamp, starts + best RMSE (or `snapshot`), the ZIR misfit (`mf=`) and
  the note; a leading `+` marks runs currently overlaid. Select a run and
  click **Load selected run** for a full replay — parameters, K_spec, BOUNDS,
  settings, temperature mode and the stored family are restored exactly as
  they were.
* **Add to overlay / Clear overlays**: overlay the selected run's target
  curve on the target-curve panels (dashed, semi-transparent, one colour per run)
  to compare target iterations side by side; the active curve stays solid.
* **Snapshot current as run** (Project panel): records the current parameter
  set into the run history *without* a multistart (shows as `snapshot`) — use
  it to checkpoint an eye-fit before changing course.
* Opening a project restores everything (including fold state, log-x and
  temperature mode) and replays the latest run automatically. Projects saved
  by older GUI versions open unchanged (the new manifest keys are optional).

Size is negligible: a project with embedded data plus a 400-start run is a few
tens of KB; even ~100 accumulated runs stay in the low-MB range. Keep
`.otproj` files in your data tree, not in the git repo.

Keep run outputs in your data tree, not in the git repo — data and
figures are git-ignored by design.

---

## 8. Methodology caveats (read before reporting numbers)

* **The multistart fits the hand-set target curve, not the observed scatter.**
  The accepted family quantifies *parameter non-uniqueness around your target*;
  its width is conditional on that target and on the BOUNDS + acceptance
  multiplier. **Do not report the single-target family width as the
  melt-volatile uncertainty** — use the multi-target envelope procedure
  (`inversion/ot_target_envelope.py`).
* **H₂O is only weakly constrained by apatite Cl-OH-F.** `H2O_Sat` (and hence
  initial H₂O) trades off against depth/solubility assumptions; expect the
  family to spread across the full `H2O_Sat` bounds. Treat saturation as an
  external petrological constraint.
* A **flat objective** (nearly all starts accepted) means the family is
  reproducing your bounds, not resolving parameters — check the rail flags and
  the RMSE histogram.
* Temperature: prefer a real T-F curve over a constant T; the partitioning is
  T-dependent.
* Viscosity preview: PLACEHOLDER majors; melt (not magma) viscosity; H₂O
  beyond ~8 wt% extrapolates GRD08; isobaric.

---

## 9. Command-line equivalents

The GUI is a front end; every step also runs headless:

| step                    | CLI                                              |
|-------------------------|--------------------------------------------------|
| forward + golden check  | `python apatite_model.py --tf <curve> --golden <Tgt.csv>` |
| create-target + multistart | `python apatite_inversion.py both`            |
| OT production run       | `python inversion/ot_inversion_run.py`           |
| coupled viscosity figure| `python ot_viscosity_volatile_overlay.py`        |
| GRD08 validation        | `python validate_grd08.py`                       |
