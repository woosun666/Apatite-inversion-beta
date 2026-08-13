# apa-inversion-beta — Ok Tedi apatite data processing & volatile-inversion tool

Standalone Windows desktop program (no Python install required). Copy the whole
`apa-inversion-beta` folder to any Windows machine and double-click
**`apa-inversion-beta.exe`**.

> Engine ported from the apatite volatile model of Lormand et al. (2024,
> *Lithos*; built on Humphreys et al. 2021); viscosity is Giordano et al.
> (2008) GRD08; anion-site recalculation follows the ApThermo (Li & Costa
> 2020) 26-oxygen stoichiometry protocol. Please cite the original papers.

## Quick start

1. **Launch** `apa-inversion-beta.exe` → after the splash screen a project
   dialog appears: pick a recent project / *Browse* for an `.otproj` /
   **New blank session**.
2. **Two workflows**, switchable at any time and coexisting in one project:
   - **Apatite data processing** (tab *Apatite data (M1)*): `Import EPMA`
     loads CSV/XLSX (column-mapping dialog with sensible EPMA-export
     defaults) → `Recalculate` gives X_F/X_Cl/X_OH, ratios and quality flags
     → select rows in the table to tag ZHAI/matrix, suite and alteration →
     three buttons open ternary / volatile-ratio / custom plot windows.
   - **Volatile inversion** (*Explore target curve* + *Multistart* tabs):
     load the observed-ratio CSV and a T–F cooling curve in the left panel
     (or generate one from the Huber-formula b value) → drag the sliders
     until the curve passes through the data cloud → set BOUNDS and starts on
     the Multistart tab → `Run multistart`. The objective defaults to
     **logrmse** (`rmse` selectable; the two are not comparable, as labelled
     in the UI). `RMSE diagnostics window` shows parameter sensitivity;
     `Envelope (3 targets)` runs the low/central/high target union = an
     honest melt-volatile range.
3. **Save**: all data, parameters, run history and figures go into a single
   `.otproj` file (prompted before closing; a crash-recovery copy
   `.autosave~` is written every 5 minutes).
4. **Figures**: every pop-out figure window has axis-range controls and a
   Save button (SVG/PDF/300-dpi PNG); vector copies are archived into the
   project's `figures/`.

## Notes

- The multistart family width is **parameter trade-off**, not data
  uncertainty; report ranges from the Envelope union. See the bundled
  `GUI_MANUAL.md` for the methodology caveats (the in-app labels state the
  same).
- Initial volatiles and the saturation threshold are **inversion outputs**;
  the cooling schedule (T_initial/T_final/b) is a given input.
- **Melt S output location**: the single-file build writes to
  `%USERPROFILE%\apa-inversion-beta-output\apatite_smelt_out` by default
  (changeable in the tab; the `OT_SMELT_OUT` / `OT_OUT` env vars override).
  It never writes into the `%TEMP%` extraction dir, which is deleted on exit.
- First launch is a few seconds slower than the script version (runtime
  unpacking) — this is normal.
- If antivirus software flags the exe: a routine PyInstaller false positive;
  add the folder to the allow-list.

## Folder contents

```
apa-inversion-beta.exe     main program
_internal/                 runtime (do not touch; includes the full GUI_MANUAL.md and splash)
README.md                  this file
```
