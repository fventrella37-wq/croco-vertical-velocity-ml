# croco-vertical-velocity-ml

Python code, trained models and results for predicting the vertical ocean velocity $w$ from
the near-surface horizontal velocity $(u, v)$ in output of the CROCO ocean model.

The vertical velocity at 16 fixed depths between −11.2 and −116.6 m is predicted from a
$7\times7$ stencil of $(u, v)$ at −11.2 m with two models:

- a per-level linear regression (one fit for each target depth);
- a convolutional neural network conditioned on depth through Feature-wise Linear Modulation
  (FiLM), in PyTorch.

The models are trained on one region of the Alb500 CROCO simulation of the western
Mediterranean and evaluated on two other regions.

## Repository structure

| Path | Content |
|---|---|
| `notebook_nuredduna.ipynb` | Main workflow: reading the CROCO output with xcroco, interpolation onto fixed z-levels, region selection, 3D divergence, vertical correlation of $w$, per-level linear regression, FiLM network |
| `notebook_nuredduna_seeds.ipynb` | Training of an ensemble of 274 FiLM networks with different random seeds; ensemble statistics |
| `notebook_nuredduna_noise.ipynb` | Sensitivity of both models to white Gaussian noise added to the input velocities |
| `notebook_nuredduna_temporal.ipynb`, `temporal_validation.py` | Evaluation of the models on other snapshots of the simulation and training on summer snapshots (the `.py` file is the script version of the notebook) |
| `scripts/` | Plotting scripts |
| `cnn_seeds_film/` | Trained FiLM networks, one folder per seed: `config.json`, `model_final.pt` (PyTorch `state_dict`), `history.csv`, `result.json`, validation $R^2$ (`*_r2.csv`); ensemble summaries (`ensemble_*.csv`, `lr_reference.*`); `noise_algerian/` (noise experiment) |
| `lr_perlevel_run_20261008_174912/` | Per-level linear regression: fitted model (`.pkl`, scikit-learn) and out-of-sample $R^2$ at the 16 target levels |
| `temporal_validation/`, `temporal_validation_summer_2136/`, `temporal_validation_summer_2360/` | Results of the evaluation on other snapshots (CSV) |
| `cnn_summer_film_2136/`, `cnn_summer_film_2360/` | FiLM networks trained on the snapshots of 7 August and 4 September 2011 |

The notebooks are distributed without their outputs. The result folders (`cnn_seeds_film/`,
`lr_perlevel_run_20261008_174912/`, `temporal_validation*/`, `cnn_summer_film_*/`) are distributed
as zip archives with the same names: extract them in the repository root before use (see Installation).

### Region names

| Name in code and files | Region |
|---|---|
| `Ibiza` | South-West Mediterranean (training) |
| `Alboran` | Alboran Sea (evaluation) |
| `Algerian` | transition region between the Alboran Sea and the open western Mediterranean (evaluation) |

## Input data (not included)

The code reads the output of the Alb500 CROCO simulation (500 m horizontal resolution,
70 σ levels, 3-hourly averages) of Capó et al. (2021), which is not part of this repository:

- `grd_alb500sm.nc`: model grid;
- `croco_avg.0656.nc`: training snapshot (time index 0, 3 February 2011, 19:30 UTC);
- other `croco_avg.*.nc` files of 2011 for the evaluation on other snapshots, including
  `croco_avg.2136.nc` and `croco_avg.2360.nc` (summer training snapshots).

The files are expected in the repository root (they are ignored by git).

Capó, E., McWilliams, J. C., Mason, E., & Orfila, A. (2021). Intermittent frontogenesis in the
Alboran Sea. *Journal of Physical Oceanography*, 51(5), 1417–1439.
https://doi.org/10.1175/JPO-D-20-0277.1

## Installation

```bash
git clone <repository URL>
cd croco-vertical-velocity-ml
for f in *.zip; do unzip -q "$f"; done   # extracts the result folders
pip install -r requirements.txt
```

`xcroco` is part of the CROCO python tools **croco_pytools v2.0.4**, distributed by the CROCO
project and not on PyPI: unpack it as `croco_pytools-v2.0.4/` in the repository root (the
notebooks add `./croco_pytools-v2.0.4/xcroco` to the Python path). Its dependencies are listed
in `croco_pytools-v2.0.4/env.yml`.

## Usage

Run the notebooks and `temporal_validation.py` from the repository root, in this order:

1. `notebook_nuredduna.ipynb` (writes `derivate_oceaniche.npz`, the fitted linear regression
   and a single FiLM network);
2. `notebook_nuredduna_seeds.ipynb` (writes `cnn_seeds_film/`);
3. `notebook_nuredduna_noise.ipynb` (writes `cnn_seeds_film/noise_algerian/`);
4. `notebook_nuredduna_temporal.ipynb`, or `python -u temporal_validation.py` (writes
   `temporal_validation*/` and `cnn_summer_film_*/`).

The trained networks and the results of steps 2–4 are already included.

### Plotting scripts

The scripts read the CROCO files and the result folders from the repository root, write the
figures to `figures_out/` and intermediate files to `cache/`. The three folders can be changed
with the environment variables `VV_WORKDIR`, `VV_FIGDIR` and `VV_CACHE` (see `scripts/paths.py`).

| Script | Output | Needs |
|---|---|---|
| `fig_zlevels_domains.py` | fixed z-levels and map of the three regions | grid, `croco_avg.0656.nc` |
| `fig_arch_agu_v5.py out.svg out.pdf` | diagram of the FiLM network | TeX Gyre Heros and Latin Modern Math fonts (paths at the top of the script), `cairosvg` |
| `fig_divergence.py` | probability density function of the 3D divergence | `derivate_oceaniche.npz` |
| `prep2.py`, then `runcells_agu.py <matrix\|corr\|lrcoef> <Q>` | correlation matrix of $w$, correlation with the reference level, regression coefficients (`Q`: scale factor of the figure size) | `croco_avg.0656.nc`, grid |
| `prep.py`, `predict.py`, then `fig_val_maps_agu.py <LR\|FiLM>` | maps of predicted and CROCO $w$ on the two evaluation regions | `croco_avg.0656.nc`, grid |
| `fig_training_agu.py` | training loss and $R^2$ of the ensemble during training | `cnn_seeds_film/` |
| `sm_temporal_figures.py` | $R^2$, boundary layer and vertical correlation of $w$ over the year | `temporal_validation*/`, grid, `croco_avg.0656.nc` |

`runcells_agu.py` runs the plotting cells of `notebook_nuredduna.ipynb`; `agu_style.py` sets
the common plotting style.

### Loading a trained network

`model_final.pt` is a PyTorch `state_dict`. The network class is defined in Section 10 of
`notebook_nuredduna.ipynb` and the parameters of each run are in its `config.json`. Inputs
and targets are standardized with statistics of the training region (computed in the notebook).

## License

Not chosen yet.
