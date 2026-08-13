# Ok Tedi apatite volatile inversion & melt-viscosity toolkit

Companion code for a manuscript (in review) on the Ok Tedi porphyry Au(–Cu)
deposit, Papua New Guinea: apatite as a recorder of magmatic volatiles
(F, Cl, H₂O, S) and the derived melt-viscosity evolution of the ore-forming
magma. The tools are generic — any apatite Cl–OH–F dataset with a T–F cooling
path can be run through them.

| Component | What it does |
|---|---|
| `apatite_model.py` / `apatite_inversion.py` | Forward apatite Cl–OH–F volatile model along a T–F crystallisation path (Python port of the Lormand et al. 2024 MATLAB model, built on Humphreys et al. 2021) + two-stage multistart inversion |
| `apatite_m1.py` | EPMA processing: ApThermo (Li & Costa 2020) 26-oxygen anion-site recalculation, quality flags, review plots |
| `apatite_smelt.py` (+ `apatite_smelt_panel.py`) | Apatite S → melt S concentration (Tassara et al. 2026 formulation) |
| `giordano2008.py` / `magma_viscosity.py` | GRD08 (Giordano et al. 2008) VFT melt viscosity; optional crystal (Roscoe) and bubble (Llewellin & Manga 2005) suspension terms |
| `build_melts_lld.py` / `magma_viscosity_evolution.py` | rhyolite-MELTS liquid line of descent (via alphaMELTS for Python) + viscosity η(F) trajectory |
| `viscosity_path.py` / `ot_viscosity_volatile_overlay.py` | Two-branch melt-viscosity cooling path across H₂O saturation; overlay version couples it to the inversion output |
| `inversion/` | Headless canonical inversion runner + 3-target envelope (honest melt-volatile uncertainty) |
| `apatite_gui.py` | Tkinter GUI wrapping all of the above — 6 tabs, single-file `.otproj` projects |

## Install

Python ≥ 3.9 (validated on 3.9 and 3.11+):

```bash
pip install -r requirements.txt
```

Optional extras:

- `sv-ttk` — GUI theme (already in `requirements.txt`; the GUI falls back to
  stock ttk without it)
- **alphaMELTS for Python** — only needed by `build_melts_lld.py`; point the
  `ALPHAMELTS_PY` env var at your checkout
- **PyInstaller ≥ 6.21** — only for building the Windows exe

## Quick start (GUI)

```bash
python make_sample_project.py   # writes sample_data/ (synthetic demo inputs)
python apatite_gui.py           # or apatite_gui.bat on Windows
```

In the GUI: *New blank session* → load `sample_data/sample_obs.csv` (observed
ratios) and `sample_data/sample_tf.csv` (T–F curve) in the left panel, and
`sample_data/sample_epma.csv` via the *Apatite data (M1)* tab.

- Full user manual: [GUI_MANUAL.md](GUI_MANUAL.md)
- GUI internals: [ARCHITECTURE.md](ARCHITECTURE.md)
- Prebuilt Windows exe manual: [README_EXE.md](README_EXE.md)

## Volatile inversion — protocol summary

Two-stage design, faithful to the MATLAB original (**not** a direct scatter
fit):

1. **Target**: hand-tune forward parameters until the curve passes through the
   observed ratio cloud (zircon-hosted apatite inclusions + matrix apatite
   combined by design — the two populations jointly bracket the path).
2. **Multistart**: bounded L-BFGS-B starts fit 9 parameters (H₂O/Cl/F init,
   3 crystal–melt D, 2 fluid–melt D, H₂O_Sat) to that target. Objective =
   **logRMSE** (RMSE of log₁₀ residuals over the XCl/XOH + XF/XOH channels;
   plain `rmse` selectable but not comparable — always label the metric).

**Reporting discipline:**

- Central value = accepted-family **median** — the single best point is
  degenerate along the H₂O_init ↔ H₂O_Sat valley.
- Family width = parameter trade-off, **never the data error bar**. Report the
  **bracket-envelope union** instead (`inversion/ot_target_envelope.py`:
  low/central/high targets, eye-validated in `preview` mode, then `full`).
- Engine invariant: the H₂O_init upper bound stays below the H₂O_Sat lower
  bound (first step undersaturated; asserted in `apatite_model`).

Headless canonical run (data dir via env; see *Data availability*):

```bash
python inversion/ot_inversion_run.py            # env: OT_DATA_DIR, NUM_START, N_JOBS
python inversion/ot_target_envelope.py preview  # tune brackets on the cloud
python inversion/ot_target_envelope.py full     # -> envelope union
```

Generic entry point for non–Ok Tedi data: edit the USER-EDITABLE block of
`apatite_inversion.py` (`--jobs` parallelises), or just use the GUI.

## Viscosity products

Melt viscosity = Giordano, Russell & Dingwell (2008) VFT; coefficients verified
against the paper (`python validate_grd08.py`). H₂O and F are the volatile
inputs (Cl and S have no GRD08 term). H₂O contents near/above the ~8 wt%
calibration edge are extrapolations — declare them when publishing.

```bash
python build_melts_lld.py                # LLD majors library (needs alphaMELTS
                                         # + a compositions CSV, see OT_BULK)
python viscosity_path.py                 # two-branch cooling path (demo majors)
python viscosity_path.py --validate      # self-check
python ot_viscosity_volatile_overlay.py  # inversion-coupled overlay + panels
                                         # (needs OT_DATA_DIR + OT_LLD)
python apatite_smelt.py                  # melt S pipeline (needs OT_SMELT_* inputs)
```

The LLD is built with rhyolite-MELTS 1.0.2, fractional crystallisation,
isobaric (200 MPa default), fO₂ on an FMQ-offset buffer; total Fe enters as
Fe₂O₃ and the buffer sets the ferric/ferrous split (Kress & Carmichael 1991).
The initial melt H₂O of the LLD is a build parameter — declare it separately
from the inverted H₂O when publishing.

## Data availability

**No measurement data ship with this repository — it is code-only.** The Ok
Tedi EPMA data, whole-rock compositions, cooling curve and inversion products
are provided with the manuscript's supplementary material; point the
environment variables below at wherever you unpack them. The GUI runs
self-contained on the synthetic `sample_data/`.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `OT_DATA_DIR` | `inversion/*`, overlay | run dir holding the observed-ratio CSV + T–F curve, and the inversion products |
| `OT_LLD` | overlay | LLD majors CSV (output of `build_melts_lld.py`) |
| `OT_BULK` | `build_melts_lld.py` | whole-rock compositions CSV (`sample,SiO2,…,P2O5[,LOI][,H2O]`) |
| `OT_FIG_DIR` | `viscosity_path.py` | figure/CSV output dir (default `viscosity_path_out/`) |
| `OT_SMELT_{ZIRCON,ZHAI,MATRIX,GOLDEN,OUT}` | `apatite_smelt.py` | melt-S inputs/outputs (also settable in the GUI dialog) |
| `ALPHAMELTS_PY` | `build_melts_lld.py` | alphaMELTS-for-Python checkout (default `../alphamelts-py`) |
| `OKTEDI_REPO` | `inversion/*` | engine repo root (default: parent of `inversion/`) |
| `NUM_START`, `N_JOBS`, `OT_METRIC`, `X_MODE` | `inversion/*`, overlay | protocol knobs — see the script headers |

## Validation gates

Self-contained (run out of the box):

```bash
python validate_grd08.py          # GRD08 coefficients + worked example
python validate_gui_state.py      # byte-compile + golden compute_state + widget smoke
python viscosity_path.py --validate
python giordano2008.py            # GRD08 self-check + placeholder-majors demo
```

Data-dependent (need the supplement data): `apatite_model.py --tf <T-F csv>
--golden <MATLAB _Tgt.csv>`, `apatite_smelt.py --golden`, and the golden
multistart reproduction in `inversion/ot_inversion_run.py`.

## Windows exe

```powershell
powershell -File build_win.ps1 -OneFile
```

→ `dist/apa-inversion-beta.exe` (PyInstaller, ~90 MB). `splash.png` is a
version-controlled build input. First launch takes a few seconds longer than
the script version (runtime unpacking).

## References

- Giordano D., Russell J.K., Dingwell D.B. (2008), *EPSL* 271 — GRD08 VFT melt-viscosity model.
- Lormand C., et al. (2024), *Lithos* — apatite volatile forward model (MATLAB original).
- Humphreys M.C.S., et al. (2021), *EPSL* — apatite volatile-evolution framework.
- Huber C., Bachmann O., Manga M. (2010) — T–F cooling parameterisation (b-exponent curves).
- Li W., Costa F. (2020) — ApThermo apatite anion-site recalculation.
- Tassara S., et al. (2026) — apatite S → melt S conversion; S partitioning terms after Konecke et al. (2019) and Parat & Holtz (2004).
- Gualda G.A.R., et al. (2012), *J. Petrol.* and Ghiorso M.S., Gualda G.A.R. (2015) — rhyolite-MELTS; alphaMELTS for Python (Antoshechkina & Ghiorso).
- Kress V.C., Carmichael I.S.E. (1991) — Fe³⁺/Fe²⁺–fO₂ relation.
- Roscoe R. (1952); Llewellin E.W., Manga M. (2005) — crystal / bubble suspension terms.

Please verify each entry against the original publication before citing.

## License & citation

MIT License — see [LICENSE](LICENSE). If you use this code, please cite the
companion manuscript (in review; the citation will be added here on
publication) together with the original model papers above. The third-party
MATLAB model this engine was ported from is **not** redistributed here.
