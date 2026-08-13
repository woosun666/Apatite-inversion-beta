# Ok Tedi — bulk magma viscosity along the crystallisation path, η(F)

`magma_viscosity_evolution.py` computes how the **bulk viscosity of the Ok Tedi
porphyry Au(–Cu) ore-forming magma evolves through crystallisation**, η(F), by
coupling a rhyolite-MELTS liquid line of descent with an apatite-based volatile
model. It extends the static calculator [`magma_viscosity.py`](magma_viscosity.py)
from a single point to a property *trajectory*.

> Viscosity is reported as a derived **intensive property** ("low-viscosity,
> hydrous, mobile magma") — **not** an ascent / transport / flux argument.

## Method

At every melt fraction `F`:

```
η_magma(F) = η_melt(GRD08) × η_crystal(Roscoe) × η_bubble(Llewellin & Manga 2005)
```

with the evolving inputs supplied by:

| input | source |
|---|---|
| major oxides(F), T(F) | rhyolite-MELTS **liquid line of descent (LLD)** |
| melt H₂O(F), F(F)     | apatite / Lormand volatile model (Rayleigh + fluid exsolution; primary boiling off) |
| crystal φ(F) = 1 − F  | melt-fraction complement; Roscoe lock-up at φ_max ≈ 0.6 |
| bubble φ_b(F)         | 0 while undersaturated; fixed (e.g. 0.10) once H₂O-saturated |

Only **H₂O and F** enter the melt-viscosity model (GRD08); Cl and S are not
viscosity parameters.

## Usage

Pure Python 3 standard library (`math`, `csv`); `matplotlib` optional (plot only).

```bash
# 1. build the rhyolite-MELTS LLD for sample 24OT02-57
python build_melts_lld.py                                  # -> oktedi_melts_lld.csv

# 2. compute the viscosity trajectory
python magma_viscosity_evolution.py oktedi_melts_lld.csv
```

**Outputs** (`<lld-stem>__viscosity.*`):
- `…__viscosity.csv` — per-F table: T, SiO₂, melt H₂O/F, φ, saturation & lock-up
  flags, `log η` (melt / +crystals / +bubbles).
- `…__viscosity.png` — η vs F plot (melt, bulk, +bubbles; H₂O-saturation and
  φ = 0.5 markers).

`example_melts_lld.csv` is an **illustrative placeholder only** — not Ok Tedi and
not reportable.

## Example result (24OT02-57, fractional LLD)

Liquidus ≈ 1040 °C; mobile window **F ≥ 0.5 (φ ≤ 0.5): bulk log₁₀(η / Pa·s) ≈
2.4 – 5.2**; rheological lock-up at φ ≈ 0.6 (F ≈ 0.40); melt H₂O saturates at
F ≈ 0.42. → a water-rich, low-viscosity (10³–10⁵ Pa·s), mobile intermediate magma.

## Caveats

- **GRD08 coefficients are unverified** against Giordano et al. (2008) Table 1 —
  results are provisional until checked.
- rhyolite-MELTS lacks amphibole/biotite → the LLD is biased for this hydrous,
  oxidized magma; cross-check with Petrolog3.
- φ is taken as 1 − F (not independently measured); P, fO₂, and initial H₂O are
  model defaults → run a sensitivity grid before reporting.

## References

Giordano, Russell & Dingwell (2008) *EPSL* (GRD08); Roscoe (1952); Llewellin &
Manga (2005) *JVGR*; Humphreys et al. (2021); Lormand et al. (2024); Gualda et al.
(2012) rhyolite-MELTS.
