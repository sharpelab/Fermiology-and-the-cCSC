# Fermiology and the chiral superconducting state

Figure-generation code for *"Fermiology of the chiral superconducting state in
tetralayer rhombohedral graphene"*.

Each figure in the main text, Extended Data, and Supplementary Information has
its own folder containing a self-contained Jupyter notebook that regenerates
that figure from the deposited data.

## Repository layout

```
fig1/ … fig4/            main-text figures
ed-fig-*/                Extended Data figures
si-fig-*/                Supplementary figures
fig-theory-LL-qo-fts/    theory Landau-level QO supplement
utils/                   shared library
  utils.py               analysis + plotting helpers (FFTs, detrending, Onsager/Streda)
  Lf_analysis.py         low-field SdH analysis class
  fft_sup_imports.py     common imports + plotting style
  sweep/                 read-only subset of the lab data loader (numpy only)
rmg19.json               device gate-calibration constants
requirements.txt
data/                    NOT in git — download from the Stanford Digital
                         Repository (see "Data" below) and unzip here
```

Every notebook begins with a one-line bootstrap that puts `utils/` on the path:

```python
import sys, os
sys.path.insert(0, os.path.abspath("../utils"))
```

so notebooks are meant to be run **from within their own figure folder**.

## Data

The code is published without data. The full dataset is deposited in the
**Stanford Digital Repository** (DOI: _TBD_). Download the archive and unpack it
so the repository contains a top-level `data/` directory:

```
data/
  raw_sweeps/<id>/       raw transport sweeps (metadata.json + data.tsv.gz),
                         loaded by `sweep.sweep_load.pload`
  fig1/                  band-structure / DOS theory (.h5, .npz) + DOS csv
  fig3/                  Landau-level spectra (.h5), Onsager curves, m* fits
  ed-fig-large-nD-map/   DOS map (.h5)
  ed-fig-optical/        optical image + AFM .tiff scans
```

Notebook data paths are relative (`../data/...`), so no editing is needed once
`data/` is in place.

## Setup

```bash
uv venv                       # or: python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# The ed-fig-optical AFM panel additionally needs `pspylib` (a lab .tiff
# reader), which is not bundled; obtain it separately or read the scans with
# `tifffile`.
```

Then launch Jupyter, open the notebook inside any figure folder, and run all
cells.

## Notes

- `utils/sweep/` is a minimal, read-only subset of the Sharpe-lab `measureme`
  package (just `sweep_load` and `raster`), vendored so the repository has no
  dependency on the lab acquisition stack. It depends only on `numpy`.
- `rmg19.json` holds the device gate-calibration constants used to convert
  gate voltages to carrier density `n` and displacement field `D`.
