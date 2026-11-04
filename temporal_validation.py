#!/usr/bin/env python
# Temporal validation, script version of notebook_nuredduna_temporal.ipynb
# (same code cells, in order). Run from the project folder on the server:
#     python -u temporal_validation.py > temporal_validation.log 2>&1
# Output in ./temporal_validation/ (resumable: done snapshots are skipped).
import matplotlib
matplotlib.use('Agg')            # no display: figures are only saved to file

# ===================== notebook cell 1 =====================
import sys, os, warnings, glob, gc, json, csv, time
# HDF5 file locking can make netCDF4 files on network/shared disks (e.g. /data)
# hang on open: disable it before netCDF/xarray are imported.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# Local xcroco package (CROCO python tools)
XCROCO_PATH = os.path.expanduser('./croco_pytools-v2.0.4/xcroco')
if XCROCO_PATH not in sys.path:
    sys.path.insert(0, XCROCO_PATH)

import inout as io      # I/O helpers to open CROCO grid + history/average files
import gridop as gop    # grid operations: staggering, rotation, vertical interpolation
from model import Model # native CROCO grid conventions

print('Setup complete')

# ===================== notebook cell 2 =====================
# ===== PATHS =====
DATA_DIR  = os.path.expanduser('./')
AVG_FILE  = os.path.join(DATA_DIR, 'croco_avg.0656.nc')   # training day (CROCO time-averaged output)
GRID_FILE = os.path.join(DATA_DIR, 'grd_alb500sm.nc')     # CROCO grid file (curvilinear, ~500 m)
T_IDX     = 0                                             # training snapshot within AVG_FILE
ROTATE    = False                                         # same flag as the training notebook (Section 4)

# ===== REGION BOXES (same lat/lon limits as the training notebook) =====
BOXES = {
    'Train':      (36.90, 38.30,  0.40,  2.00),   # (lat_min, lat_max, lon_min, lon_max)
    'Alboran':    (35.70, 36.40, -3.60, -2.20),
    'Transition': (36.40, 37.40, -1.50, -0.20),
}
print('Parameters defined')

# ===================== notebook cell 3 =====================
# ============================================================
# Fixed z-levels, region boxes and fields_at(): u, v, w on the fixed
# z-levels, computed ONLY inside the region boxes. Same operations as
# Sections 3-7 of the training notebook (get_z, to_rho, isoslice for u,v,
# column-by-column interp1d for w, flip k=0 surface, box crop and mask),
# without keeping any full-domain array in memory.
# ============================================================
croco = Model('croco_native')
_open = {}

def _open_file(path):
    if path not in _open:
        _open.clear()                                   # keep only the file in use open
        _open[path] = io.open_files(croco, GRID_FILE, [path], grid_metrics=1, chunks={'t': 1})
    return _open[path]

# --- Z_LEVELS: native sigma levels of the deepest column at the training snapshot (Section 3)
_ds1 = _open_file(AVG_FILE).isel(t=T_IDX)
_xg1 = gop.fast_xgcm_grid(_ds1, croco, grid_metrics=1)
_zr1 = gop.get_z(croco, ds=_ds1, z_sfc=_ds1.z_sfc, xgrid=_xg1, vgrid='r', hgrid='r')
_h = _ds1.h.values
j_deep, i_deep = np.unravel_index(np.nanargmax(_h), _h.shape)
Z_ASC = np.sort(np.asarray(_zr1[:, j_deep, i_deep].values))   # bottom -> surface
Z_LEVELS = Z_ASC[::-1].copy()                                # k = 0 surface (Section 6)
lon, lat = _ds1.lon.values, _ds1.lat.values
del _zr1, _xg1, _ds1
print(f'Z_LEVELS: {len(Z_LEVELS)} levels, {Z_LEVELS[0]:.1f} m (k=0) to {Z_LEVELS[-1]:.1f} m')

# --- region boxes: index ranges + mask of the points outside the lat/lon range (Section 7)
_BOX = {}
for reg, (la0, la1, lo0, lo1) in BOXES.items():
    m = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    jj, ii = np.where(m)
    j0, j1, i0, i1 = jj.min(), jj.max(), ii.min(), ii.max()
    _BOX[reg] = dict(ys=slice(j0, j1 + 1), xs=slice(i0, i1 + 1), mask=~m[j0:j1 + 1, i0:i1 + 1],
                     lat=lat[j0:j1 + 1, i0:i1 + 1], lon=lon[j0:j1 + 1, i0:i1 + 1])
    print(f'{reg:<11} box {j1 - j0 + 1} x {i1 - i0 + 1}, valid points {m.sum()}')


VERBOSE = True   # print the time of every step of fields_at (to locate slow/hanging steps)

def _step(msg, t0):
    if VERBOSE:
        print(f'    [{time.time() - t0:6.1f} s] {msg}', flush=True)

def fields_at(path, t, regions):
    """u, v, w on Z_LEVELS (k=0 surface) inside the given region boxes, at
    time index t of `path`."""
    _ts = time.time()
    _step(f'open {os.path.basename(path)}, t={t}', _ts)
    d1 = _open_file(path).isel(t=t)
    xg = gop.fast_xgcm_grid(d1, croco, grid_metrics=1)
    zr = gop.get_z(croco, ds=d1, z_sfc=d1.z_sfc, xgrid=xg, vgrid='r', hgrid='r')
    zw = gop.get_z(croco, ds=d1, z_sfc=d1.z_sfc, xgrid=xg, vgrid='w', hgrid='r')
    _step('grid and z built (lazy)', _ts)
    if ROTATE:
        ur, vr = gop.rot_uv(d1.xcur.compute(), d1.ycur.compute(), d1.angle, xg)
        ug, vg = gop.to_rho(ur.compute(), xg), gop.to_rho(vr.compute(), xg)
    else:
        ug, vg = gop.to_rho(d1.xcur.compute(), xg), gop.to_rho(d1.ycur.compute(), xg)
    _step('u, v on rho points', _ts)
    zrc = zr.compute()
    _step('z_rho computed', _ts)
    uz = gop.isoslice(ug, Z_ASC, xg, target_data=zrc, axis='z').isel(z_r=slice(None, None, -1))
    vz = gop.isoslice(vg, Z_ASC, xg, target_data=zrc, axis='z').isel(z_r=slice(None, None, -1))
    _step('u, v isoslice set up', _ts)
    yd, xd = d1.zcur.dims[-2:]
    out = {}
    for reg in regions:
        b = _BOX[reg]
        U = uz.isel(y=b['ys'], x=b['xs']).compute().values
        V = vz.isel(y=b['ys'], x=b['xs']).compute().values
        zc  = d1.zcur.isel({yd: b['ys'], xd: b['xs']}).values
        zwb = zw.isel({zw.dims[-2]: b['ys'], zw.dims[-1]: b['xs']}).values
        W = np.full((len(Z_ASC),) + zc.shape[1:], np.nan)
        for jj in range(zc.shape[1]):
            for ii in range(zc.shape[2]):
                cz, cw = zwb[:, jj, ii], zc[:, jj, ii]
                ok = np.isfinite(cz) & np.isfinite(cw)
                if ok.sum() >= 2:
                    W[:, jj, ii] = interp1d(cz[ok], cw[ok], bounds_error=False, fill_value=np.nan)(Z_ASC)
        W = W[::-1]
        for A in (U, V, W):
            A[:, b['mask']] = np.nan
        out[reg] = (U, V, W)
        _step(f'{reg}: u, v, w in the box', _ts)
    del uz, vz, ug, vg, zr, zrc, zw, xg, d1
    gc.collect()
    return out


# --- training day, the three boxes (input of the FiLM setup cell below)
_t0 = time.time()
TRAIN_FIELDS = fields_at(AVG_FILE, T_IDX, ['Train', 'Alboran', 'Transition'])
U_IB, V_IB, W_IB = TRAIN_FIELDS['Train']
U_NA, V_NA, W_NA = TRAIN_FIELDS['Alboran']       # internal names as in the training notebook
U_AG, V_AG, W_AG = TRAIN_FIELDS['Transition']
lat_NA, lon_NA = _BOX['Alboran']['lat'], _BOX['Alboran']['lon']
lat_AG, lon_AG = _BOX['Transition']['lat'], _BOX['Transition']['lon']
print(f'Training day processed in {time.time() - _t0:.0f} s: '
      f'U_IB {U_IB.shape}, U_NA {U_NA.shape}, U_AG {U_AG.shape}')

# ===================== notebook cell 4 =====================
# ============================================================
# FiLM ENSEMBLE — SETUP (run once): parameters, model/helpers, patch
# extraction and standardization. Identical to the FiLM cell of
# notebook_nuredduna.ipynb; the seed-dependent part (training +
# evaluation) is in run_seed() below.
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
import sys
import json
import csv
import itertools
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import r2_score
from matplotlib.colors import SymLogNorm
# ============================================================
# Requires already loaded upstream (from the preprocessing block):
#   U_IB, V_IB, W_IB   -> (nz, ny, nx)  Ibiza fields (training)
#   U_NA, V_NA, W_NA   -> (nz, ny_na, nx_na)  Alboran fields (validation, REFERENCE)
#   lat_NA, lon_NA     -> 2D Alboran grids for plotting
#   U_AG, V_AG, W_AG   -> (nz, ny_ag, nx_ag)  Algerian fields (2nd validation)
#   lat_AG, lon_AG     -> 2D Algerian grids for plotting
#   Z_LEVELS           -> real depths (meters) per level
# ============================================================
import random
import subprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ============================================================
# PARAMETERS — edit only here
# ============================================================
r_cnn          = 3         # HORIZONTAL stencil radius: patch = (2r+1) x (2r+1)
K_START_cnn    = 2         # level (index) of u,v used as input — e.g. surface/near-surface
N_TARGET_cnn   = 16        # levels 2..17 -> z down to about -116.6 m, stays above -150 m
K_PRED_cnn     = [2,7,12,17]   # absolute levels used for metrics/plots on Alboran and Algerian
ROW_BATCH_cnn   = 2048      # target number of (patch, level) ROWS per training batch: batches are
                            # built from whole patches, PATCH_BATCH_cnn = ROW_BATCH_cnn // N_TARGET_cnn
                            # patches with all their levels (conv1 once per patch, see forward_grouped)
PRED_BATCH_cnn  = 512       # batching for validation/prediction
K_EXTRAPOLATE_cnn = 20        # modest extrapolation just past training range (was 35)
N_EXTRAP_FIT_cnn  = 4         # deepest training levels used for the least-squares fit of mean_w(z)
                              # (linear) and log std_w(z) (linear) when predicting BELOW the training range
VALIDATE_EVERY_EPOCHS = 25      # cadence once training has settled
# Denser validation at the start, to resolve how val R2 rises: list of
# (up to epoch, check every N epochs); after the last entry the cadence
# is VALIDATE_EVERY_EPOCHS.
VALIDATE_SCHEDULE = [(25, 1)]

# Regularization knobs -- grouped here too (not just architecture/training
# params) so config.json is fully self-describing: no need to go read the
# model class / optimizer code to know how a run was
# regularized.
DROPOUT2D_P         = 0.1   # spatial dropout on conv feature maps (history: 0.1 -> 0.2 -> 0.1)
FC_DROPOUT_P        = 0.2   # dropout in the FC head (history: 0.3 -> 0.2)
WD_OTHER            = 1e-3  # weight_decay on conv1/conv2/FC weights (history: 1e-4 -> 1e-3 -> 3e-3 -> 1e-3)
WD_FILM             = 5e-5  # weight_decay (L2) on the FiLM generator only, see optimizer comment (history: 0.0 -> 1e-5 -> 5e-5 -> 1e-5 -> 5e-5)
# Stopping: NO fixed number of epochs. Training stops at the FIRST of
#  (a) the lr reaches LR_MIN_cnn: ReduceLROnPlateau (on the TRAIN loss) has
#      exhausted its reductions -> training has converged;
#  (b) the mean val R² on Alboran (K_PRED_cnn levels) has not improved for
#      R2_PATIENCE_EPOCHS_cnn epochs.
# Alboran only decides WHEN to stop: all evaluation uses the model at the
# stopping epoch (the final model), not the best-on-Alboran checkpoint.
LR_MIN_cnn             = 1e-6  # min lr of ReduceLROnPlateau = convergence criterion (a)
R2_PATIENCE_EPOCHS_cnn = 300   # criterion (b), in epochs (was 250 = 10 checks x 25)

# Save the run config alongside the results, so a results.npz/log.txt
# found later is self-describing without needing the notebook.
_config = dict(r_cnn=r_cnn, K_START_cnn=K_START_cnn, N_TARGET_cnn=N_TARGET_cnn,
                K_PRED_cnn=K_PRED_cnn,
                ROW_BATCH_cnn=ROW_BATCH_cnn, PRED_BATCH_cnn=PRED_BATCH_cnn,
                K_EXTRAPOLATE_cnn=K_EXTRAPOLATE_cnn, N_EXTRAP_FIT_cnn=N_EXTRAP_FIT_cnn,
                VALIDATE_EVERY_EPOCHS=VALIDATE_EVERY_EPOCHS,
                validate_schedule=VALIDATE_SCHEDULE,
                dropout2d_p=DROPOUT2D_P, fc_dropout_p=FC_DROPOUT_P,
                weight_decay_other=WD_OTHER, weight_decay_film=WD_FILM,
                lr_min=LR_MIN_cnn, r2_patience_epochs=R2_PATIENCE_EPOCHS_cnn,
                evaluated_model='final', device=str(device))
# (_config is written to config.json in every seed folder, together with the seed)


def extract_patches_with_all_levels(u_field, v_field, W_levels_full, r):
    """Extracts all (2r+1)x(2r+1) patches whose u,v neighbourhood is fully
    valid, together with the w vector at ALL target levels (even if some
    levels are NaN — they get filtered afterwards, row by row).
    Returns: patches (n_patches,2,ss,ss), y_all_levels (n_patches,n_levels)
             (can contain NaN), centers_j, centers_i (n_patches,) — indices
             in the original field.
    """
    ny, nx = u_field.shape
    ss = 2 * r + 1
    u_win = sliding_window_view(u_field, (ss, ss))
    v_win = sliding_window_view(v_field, (ss, ss))
    patch_valid = ~np.isnan(u_win).any(axis=(2, 3)) & ~np.isnan(v_win).any(axis=(2, 3))
    jj, ii = np.where(patch_valid)
    patches = np.stack([u_win[jj, ii], v_win[jj, ii]], axis=1)
    centers_j, centers_i = jj + r, ii + r
    y_all_levels = W_levels_full[:, centers_j, centers_i].T
    return patches, y_all_levels, centers_j, centers_i


def build_flat_training_rows(y_all_levels, z_of_level):
    """Expands (n_patches, n_levels) into independent (patch_idx, z, y) rows,
    dropping ONLY the (patch,level) combinations where w is NaN."""
    valid = ~np.isnan(y_all_levels)
    patch_idx, level_idx = np.where(valid)
    y_flat = y_all_levels[patch_idx, level_idx].astype(np.float32)
    z_flat = z_of_level[level_idx].astype(np.float32)
    return patch_idx, level_idx, z_flat, y_flat


class FiLM(nn.Module):
    """Generates gamma, beta from z and modulates a (B,C,H,W) tensor channel-wise:
    output = gamma * x + beta, applied identically over the whole spatial window."""
    def __init__(self, z_dim, num_channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(z_dim, num_channels * 2)
        # Canonical FiLM init: gamma starts at 1, beta at 0, i.e. FiLM starts
        # as the identity. This is only the INITIAL CONDITION: gamma/beta are
        # then free to move away from it during training. Without it,
        # nn.Linear's default weights give gamma~0 at the start and conv1's
        # features would be almost zeroed out before the network has learned
        # anything.
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        with torch.no_grad():
            self.to_gamma_beta.bias[:num_channels].fill_(1.0)  # half of gamma -> 1, beta stays 0

    def forward(self, x, z):
        gamma_beta = self.to_gamma_beta(z)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ZCondStencilCNN_FiLM(nn.Module):
    """CNN conditioned on z via a single FiLM layer, applied BEFORE
    conv2 (it modulates conv1's output). conv1 sees only the raw (u,v)
    stencil (no z at all) and always extracts the same depth-invariant
    local features; FiLM then rescales/shifts those 8 feature maps as a
    function of z, and conv2 -- a fixed, non-z-dependent 8->16
    recombination -- operates on that already depth-conditioned input.
    Single entry point for z, ~7.8k total parameters."""
    def __init__(self, stencil_size, z_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)   # padding=1 -- same-size output (7x7 -> 7x7)
        self.film1 = FiLM(z_dim, 8)                               # only entry point for z -- applied BEFORE conv2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0)   # no FiLM: fixed recombination of the z-modulated input
        self.relu = nn.ReLU()
        self.spatial_dropout = nn.Dropout2d(DROPOUT2D_P)  # value set in the PARAMETERS block above
        flat_dim = 16 * (stencil_size - 2) * (stencil_size - 2)   # conv1 is now 'same' padding (no shrink); only conv2 is valid (-2)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 16), nn.ReLU(), nn.Dropout(FC_DROPOUT_P),
            nn.Linear(16, 1)
        )

    def features(self, x_img):
        """Depth-invariant part (conv1 + ReLU): depends only on the (u,v) stencil."""
        return self.relu(self.conv1(x_img))

    def head(self, h1, z):
        """z-dependent part. Dropout on conv1's output is applied HERE, i.e.
        per (patch, z) row, so every row keeps its own independent mask
        exactly as before the conv1 hoisting."""
        x = self.spatial_dropout(h1)
        x = self.film1(x, z)                                      # z enters here, BEFORE conv2
        x = self.spatial_dropout(self.relu(self.conv2(x)))        # fixed recombination of the z-modulated input
        feat = x.flatten(start_dim=1)
        return self.fc(feat).squeeze(-1)

    def forward(self, x_img, z):
        return self.head(self.features(x_img), z)

    def forward_grouped(self, x_unique, inv, z):
        """Same as forward(x_unique[inv], z), but conv1 runs once per distinct
        patch; its output is then copied to every row that shares the patch."""
        return self.head(self.features(x_unique)[inv], z)


def predict_zcond_in_batches(model, patches_tensor, patch_idx, z_flat_s, device,
                              batch_size=PRED_BATCH_cnn):
    """Mini-batch inference. Within each chunk, conv1 is computed once per
    distinct patch (forward_grouped) and shared by all rows using it."""
    model.eval()
    n = len(patch_idx)
    z_t = torch.tensor(z_flat_s, dtype=torch.float32).unsqueeze(1)
    out = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = start + batch_size
            uniq_b, inv_b = np.unique(patch_idx[start:end], return_inverse=True)
            xb = patches_tensor[uniq_b].to(device)
            inv_t = torch.as_tensor(inv_b.ravel(), dtype=torch.long, device=device)
            zb = z_t[start:end].to(device)
            out.append(model.forward_grouped(xb, inv_t, zb).cpu())
    return torch.cat(out, dim=0).numpy()


# ------------------------------------------------------------
# 1. Extraction — Ibiza (training): horizontal patches + all w levels
# ------------------------------------------------------------
u_cnn = U_IB[K_START_cnn]
v_cnn = V_IB[K_START_cnn]
W_IB_slice = W_IB[K_START_cnn : K_START_cnn + N_TARGET_cnn]
z_levels_ib = Z_LEVELS[K_START_cnn : K_START_cnn + N_TARGET_cnn]
patches_ib, y_all_ib, _, _ = extract_patches_with_all_levels(u_cnn, v_cnn, W_IB_slice, r_cnn)
patch_idx_ib, level_idx_ib, z_flat_ib, y_flat_ib = build_flat_training_rows(y_all_ib, z_levels_ib)
stencil_size_cnn = 2 * r_cnn + 1
print(f'Ibiza — stencil {stencil_size_cnn}×{stencil_size_cnn} | valid patches: {len(patches_ib)} | '
      f'rows (patch,level) for training: {len(y_flat_ib)}')
if len(y_flat_ib) == 0:
    raise RuntimeError('No valid training rows — check r_cnn/K_START_cnn')

# ------------------------------------------------------------
# 2. Standardization
# ------------------------------------------------------------
mean_u_cnn, std_u_cnn = patches_ib[:, 0].mean(), patches_ib[:, 0].std()
mean_v_cnn, std_v_cnn = patches_ib[:, 1].mean(), patches_ib[:, 1].std()
patches_ib_s = patches_ib.copy().astype(np.float32)
patches_ib_s[:, 0] = (patches_ib[:, 0] - mean_u_cnn) / std_u_cnn
patches_ib_s[:, 1] = (patches_ib[:, 1] - mean_v_cnn) / std_v_cnn
mean_z_cnn, std_z_cnn = z_levels_ib.mean(), z_levels_ib.std()
z_flat_ib_s = (z_flat_ib - mean_z_cnn) / std_z_cnn
mean_w_per_level = np.array([np.nanmean(y_all_ib[:, t]) for t in range(N_TARGET_cnn)])
std_w_per_level  = np.array([np.nanstd(y_all_ib[:, t])  for t in range(N_TARGET_cnn)]) + 1e-10
y_flat_ib_s = (y_flat_ib - mean_w_per_level[level_idx_ib]) / std_w_per_level[level_idx_ib]
patches_ib_tensor = torch.tensor(patches_ib_s, dtype=torch.float32)

# Target stats (mean_w, std_w) at an ARBITRARY depth z [m]. They are only
# defined on the training levels, so:
#  - z between two training levels -> linear interpolation in z between the
#    two nearest levels (weights by distance); exact value on a level;
#  - z below the deepest training level -> least-squares fit on the
#    N_EXTRAP_FIT_cnn deepest levels, evaluated at z: mean_w(z) linear,
#    log(std_w)(z) linear and then std_w = exp(...), so std_w stays > 0;
#  - z above the shallowest training level -> not supported (error).
_z_train = np.asarray(z_levels_ib, dtype=np.float64)
_asc = np.argsort(_z_train)                     # ascending z: deepest first

def w_stats_at_z(z):
    z = float(z)
    z_top, z_bot = _z_train.max(), _z_train.min()
    if z > z_top + 1e-6:
        raise ValueError(f'z={z:.2f} m is above the shallowest training level ({z_top:.2f} m): not supported')
    if z >= z_bot - 1e-6:                        # inside the training range
        m = np.interp(z, _z_train[_asc], mean_w_per_level[_asc])
        s = np.interp(z, _z_train[_asc], std_w_per_level[_asc])
    else:                                        # below the training range
        sel = _asc[:N_EXTRAP_FIT_cnn]            # the N deepest training levels
        m = np.polyval(np.polyfit(_z_train[sel], mean_w_per_level[sel], 1), z)
        # std_w: linear fit of log(std_w) in z, then back to std_w with exp (always > 0)
        s = np.exp(np.polyval(np.polyfit(_z_train[sel], np.log(std_w_per_level[sel]), 1), z))
    return float(m), float(s)

# ------------------------------------------------------------
# 3. Extraction — Alboran (validation)
# ------------------------------------------------------------
u_na, v_na = U_NA[K_START_cnn], V_NA[K_START_cnn]
ny_na, nx_na = u_na.shape
W_NA_slice = W_NA[K_START_cnn : K_START_cnn + N_TARGET_cnn]
z_levels_na = Z_LEVELS[K_START_cnn : K_START_cnn + N_TARGET_cnn]
patches_na, y_all_na, centers_j_na, centers_i_na = extract_patches_with_all_levels(u_na, v_na, W_NA_slice, r_cnn)
patches_na_s = patches_na.copy().astype(np.float32)
patches_na_s[:, 0] = (patches_na[:, 0] - mean_u_cnn) / std_u_cnn
patches_na_s[:, 1] = (patches_na[:, 1] - mean_v_cnn) / std_v_cnn
patches_na_tensor = torch.tensor(patches_na_s, dtype=torch.float32)
print(f'Alboran — valid patches: {len(patches_na)}')

# ------------------------------------------------------------
# 3b. Extraction — Algerian (2nd independent validation zone)
# ------------------------------------------------------------
u_ag, v_ag = U_AG[K_START_cnn], V_AG[K_START_cnn]
ny_ag, nx_ag = u_ag.shape
W_AG_slice = W_AG[K_START_cnn : K_START_cnn + N_TARGET_cnn]
patches_ag, y_all_ag, centers_j_ag, centers_i_ag = extract_patches_with_all_levels(u_ag, v_ag, W_AG_slice, r_cnn)
patches_ag_s = patches_ag.copy().astype(np.float32)
patches_ag_s[:, 0] = (patches_ag[:, 0] - mean_u_cnn) / std_u_cnn
patches_ag_s[:, 1] = (patches_ag[:, 1] - mean_v_cnn) / std_v_cnn
patches_ag_tensor = torch.tensor(patches_ag_s, dtype=torch.float32)
print(f'Algerian — valid patches: {len(patches_ag)}')

# ===================== notebook cell 5 =====================
# ============================================================
# Fast reader for the validation snapshots: the same fields as fields_at()
# (Sections 3-7 of the training notebook), computed with numpy reading from
# the file ONLY the region boxes (the files are chunked, so this is fast and
# needs little memory; xcroco/xgcm instead load and interpolate the whole
# domain, which on /data takes minutes per snapshot and exhausts the memory).
#   z of the sigma levels: CROCO Vtransform=2 (same formula as gop.get_z),
#       z = zeta + (zeta + h) (hc sc + h Cs) / (hc + h)
#   u, v on rho points: mean of the two neighbouring u (v) points (same as
#       gop.to_rho, i.e. xgcm interp, away from the domain edges)
#   vertical interpolation onto Z_LEVELS: linear, NaN outside the column
#       (as isoslice for u, v and interp1d for w)
# It is checked below against fields_at() at the training snapshot.
# ============================================================
import netCDF4

with netCDF4.Dataset(GRID_FILE) as _g:
    _H = np.asarray(_g['h'][:], dtype=np.float64)


def _interp_columns(z, f, z_new):
    """Linear interpolation of every column f(z) (axis 0) onto z_new; NaN outside."""
    nz, ny, nx = f.shape
    out = np.full((len(z_new), ny, nx), np.nan)
    for jj in range(ny):
        for ii in range(nx):
            cz, cf = z[:, jj, ii], f[:, jj, ii]
            ok = np.isfinite(cz) & np.isfinite(cf)
            if ok.sum() >= 2:
                out[:, jj, ii] = np.interp(z_new, cz[ok], cf[ok], left=np.nan, right=np.nan)
    return out


def fields_at_np(path, t, regions):
    """Same output as fields_at(path, t, regions), reading only the boxes."""
    out = {}
    with netCDF4.Dataset(path) as d:
        hc = float(d.hc)
        sc_r, Cs_r = np.asarray(d.sc_r, float), np.asarray(d.Cs_r, float)
        sc_w, Cs_w = np.asarray(d.sc_w, float), np.asarray(d.Cs_w, float)
        for reg in regions:
            b = _BOX[reg]
            j0, j1, i0, i1 = b['ys'].start, b['ys'].stop, b['xs'].start, b['xs'].stop
            assert j0 >= 1 and i0 >= 1, 'box touches the domain edge'
            zeta = np.asarray(d['zeta'][t, j0:j1, i0:i1], dtype=np.float64)
            h = _H[j0:j1, i0:i1]
            u = np.asarray(d['u'][t, :, j0:j1, i0 - 1:i1], dtype=np.float64)
            v = np.asarray(d['v'][t, :, j0 - 1:j1, i0:i1], dtype=np.float64)
            w = np.asarray(d['w'][t, :, j0:j1, i0:i1], dtype=np.float64)
            ur = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            vr = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
            zr = zeta + (zeta + h) * (hc * sc_r[:, None, None] + h * Cs_r[:, None, None]) / (hc + h)
            zw = zeta + (zeta + h) * (hc * sc_w[:, None, None] + h * Cs_w[:, None, None]) / (hc + h)
            U = _interp_columns(zr, ur, Z_ASC)[::-1]
            V = _interp_columns(zr, vr, Z_ASC)[::-1]
            W = _interp_columns(zw, w, Z_ASC)[::-1]
            for A in (U, V, W):
                A[:, b['mask']] = np.nan
            out[reg] = (U, V, W)
    return out


# --- check against fields_at() (xcroco) at the training snapshot
_t0 = time.time()
_NP = fields_at_np(AVG_FILE, T_IDX, ['Train', 'Alboran', 'Transition'])
print(f'numpy reader: training day in {time.time() - _t0:.1f} s')
_ok_all = True
for reg in ['Train', 'Alboran', 'Transition']:
    for name, A, B in zip('UVW', _NP[reg], TRAIN_FIELDS[reg]):
        k = slice(K_START_cnn, K_START_cnn + N_TARGET_cnn)          # levels used by the models
        a, bb = A[k], B[k]
        nan_diff = int((np.isnan(a) != np.isnan(bb)).sum())
        both = np.isfinite(a) & np.isfinite(bb)
        scale = np.nanmax(np.abs(bb))
        rel = float(np.max(np.abs(a[both] - bb[both])) / scale) if both.any() else 0.0
        ok = nan_diff == 0 and rel < 1e-4
        _ok_all &= ok
        print(f'  {reg:<11} {name}: max |diff|/max|field| = {rel:.1e}, NaN mismatches = {nan_diff}'
              f'{"" if ok else "   <-- DIFFERENT"}')
if not _ok_all:
    raise RuntimeError('the numpy reader does not reproduce fields_at() at the training snapshot')
print('  ✓ numpy reader reproduces fields_at() at the training snapshot')
del _NP

# ===================== notebook cell 6 =====================
# ============================================================
# TEMPORAL VALIDATION — models trained on the training day, validated at
# other days. Nothing is retrained:
#   - per-level LR refitted on the training box at the training snapshot
#     (deterministic, identical to Section 9 of the training notebook);
#   - FiLM = final weights (model_final.pt) of ONE trained seed.
# For every snapshot the fields are processed with fields_at() (above) and
# both models are evaluated on the validation regions against the true w.
# ============================================================
import torch

# ============================================================
# PARAMETERS — edit only here
# ============================================================
# Each raw file holds one day (8 three-hourly averages); the file number
# grows by 8 per day, so the lag from the training day is (N - 656)/8 days.
RAW_DIR      = '/data/geo/lamarca/ROMS500/raw'   # all CROCO avg files on the server
FILE_EVERY   = 7                                 # one raw file every 7 (about one per week), time index 0
REGIONS      = ['Alboran', 'Transition']         # validation regions
SEED_FILM    = 261                               # representative ensemble member (paper Fig. 10)
ENSEMBLE_DIR = './cnn_seeds_film'
TEMPORAL_DIR = './temporal_validation'
PRED_BATCH_T = 16 * PRED_BATCH_cnn
# Calendar date of the model initialization (CROCO stores 'time' as
# "averaged time since initialization" in seconds, without a reference
# date). Set it, e.g. '2016-09-01 00:00', to also save the calendar date of
# every snapshot; leave None to save only the model time.
MODEL_START_DATE = None
# ============================================================

assert r_cnn == 3 and K_START_cnn == 2 and N_TARGET_cnn == 16
os.makedirs(TEMPORAL_DIR, exist_ok=True)
_raw = sorted(glob.glob(os.path.join(RAW_DIR, 'croco_avg.*.nc')))
TIME_SOURCES = [(AVG_FILE, T_IDX)] + [                 # training snapshot, then one snapshot per selected day
    (f, 0) for f in _raw[::FILE_EVERY]
    if os.path.basename(f) != os.path.basename(AVG_FILE)]
print(f'{len(_raw)} raw files found in {RAW_DIR}; {len(TIME_SOURCES)} snapshots to evaluate')


def _time_seconds(path, t):
    with xr.open_dataset(path, decode_times=False) as d:
        tv = 'time' if 'time' in d.variables else 'ocean_time'
        return float(d[tv].values[t])


def _date_of(sec):
    if MODEL_START_DATE is None:
        return ''
    import pandas as pd
    return (pd.Timestamp(MODEL_START_DATE) + pd.Timedelta(seconds=sec)).strftime('%Y-%m-%d %H:%M')


_T0 = _time_seconds(AVG_FILE, T_IDX)

# ------------------------------------------------------------
# Models (no training)
# ------------------------------------------------------------
def lr_features(p):
    n = len(p)
    return np.concatenate([p[:, 0].reshape(n, -1), p[:, 1].reshape(n, -1)], axis=1)

_X_ib = lr_features(patches_ib)
lr_models = {}
for _t in range(N_TARGET_cnn):
    _ok = ~np.isnan(y_all_ib[:, _t])
    lr_models[_t] = LinearRegression().fit(_X_ib[_ok], y_all_ib[_ok, _t])

_seed_dir = os.path.join(ENSEMBLE_DIR, f'seed_{SEED_FILM:03d}')
film = ZCondStencilCNN_FiLM(stencil_size_cnn).to(device)
film.load_state_dict(torch.load(os.path.join(_seed_dir, 'model_final.pt'), map_location=device))
film.eval()
print(f'LR refitted on the training box ({len(lr_models)} levels); FiLM = seed {SEED_FILM} (model_final.pt)')

K_ALL = list(range(K_START_cnn, K_START_cnn + N_TARGET_cnn))
_z_tr = Z_LEVELS[K_START_cnn:K_START_cnn + N_TARGET_cnn]


def evaluate(U, V, W):
    """R² of LR and FiLM at all 16 target levels for one region and one time."""
    Ws = W[K_START_cnn:K_START_cnn + N_TARGET_cnn]
    p, y_all, _, _ = extract_patches_with_all_levels(U[K_START_cnn], V[K_START_cnn], Ws, r_cnn)
    X = lr_features(p)
    r_lr = {}
    for t in range(N_TARGET_cnn):
        ok = ~np.isnan(y_all[:, t])
        r_lr[t] = float(r2_score(y_all[ok, t], lr_models[t].predict(X[ok])))
    ps = p.astype(np.float32).copy()
    ps[:, 0] = (p[:, 0] - mean_u_cnn) / std_u_cnn
    ps[:, 1] = (p[:, 1] - mean_v_cnn) / std_v_cnn
    pidx, lidx, zf, yf = build_flat_training_rows(y_all, _z_tr)
    ys = predict_zcond_in_batches(film, torch.tensor(ps), pidx, (zf - mean_z_cnn) / std_z_cnn,
                                  device, batch_size=PRED_BATCH_T)
    yp = ys * std_w_per_level[lidx] + mean_w_per_level[lidx]
    r_film = {t: float(r2_score(yf[lidx == t], yp[lidx == t])) for t in range(N_TARGET_cnn)}
    return r_lr, r_film, len(p)


# ------------------------------------------------------------
# Check at the training snapshot: R² must reproduce the stored values
# (LR: lr_reference.json; FiLM: result.json of the seed). If the lean
# processing differed from the training notebook, these would not match.
# ------------------------------------------------------------
_ref_film = json.load(open(os.path.join(_seed_dir, 'result.json')))['r2_final']
_lr_ref_path = os.path.join(ENSEMBLE_DIR, 'lr_reference.json')
_ref_lr = json.load(open(_lr_ref_path)).get('r2', {}) if os.path.exists(_lr_ref_path) else {}
_KEY = {'Alboran': 'Alboran', 'Transition': 'Algerian'}      # internal keys of the stored files
print('\nCheck at the training snapshot (computed vs stored):')
_bad = 0
for reg in REGIONS:
    r_lr, r_film, _ = evaluate(*TRAIN_FIELDS[reg])
    for k in K_PRED_cnn:
        t_ = k - K_START_cnn
        f_ref = _ref_film.get(f'{_KEY[reg]}|{k}', _ref_film.get(f'{reg}|{k}'))
        l_ref = (_ref_lr.get(_KEY[reg]) or _ref_lr.get(reg) or {}).get(str(k), {}).get('r2')
        ok_f = f_ref is None or abs(r_film[t_] - f_ref) < 1e-3
        ok_l = l_ref is None or abs(r_lr[t_] - l_ref) < 1e-3
        _bad += (not ok_f) + (not ok_l)
        print(f'  {reg:<11} k={k:<3} FiLM {r_film[t_]:.4f} (stored {f_ref if f_ref is None else round(f_ref, 4)}) '
              f'| LR {r_lr[t_]:.4f} (stored {l_ref if l_ref is None else round(l_ref, 4)})'
              f'{"" if ok_f and ok_l else "   <-- MISMATCH"}')
if _bad:
    raise RuntimeError('training-snapshot R² does not reproduce the stored values: check the processing')
print('  ✓ training snapshot reproduced')

# ------------------------------------------------------------
# Loop over time snapshots (resumable: one CSV per snapshot)
# ------------------------------------------------------------
_FIELDS = ['file', 't_idx', 'time_s', 'days_since_init', 'date', 'hours',
           'region', 'model', 'k', 'z_m', 'r2', 'n_patches']
for (path, t) in TIME_SOURCES:
    tag = f'{os.path.splitext(os.path.basename(path))[0]}_t{t:02d}'
    out_csv = os.path.join(TEMPORAL_DIR, f'r2_{tag}.csv')
    if os.path.exists(out_csv):
        print(f'{tag}: already done, skipped'); continue
    t0 = time.time()
    tsec = _time_seconds(path, t)
    hours = (tsec - _T0) / 3600.0
    tinfo = [tsec, tsec / 86400.0, _date_of(tsec), hours]
    F = TRAIN_FIELDS if (path, t) == (AVG_FILE, T_IDX) else fields_at_np(path, t, REGIONS)
    rows = []
    for reg in REGIONS:
        r_lr, r_film, n = evaluate(*F[reg])
        for t_, k in enumerate(K_ALL):
            rows.append([os.path.basename(path), t, *tinfo, reg, 'LR', k, float(Z_LEVELS[k]), r_lr[t_], n])
            rows.append([os.path.basename(path), t, *tinfo, reg, 'FiLM', k, float(Z_LEVELS[k]), r_film[t_], n])
    with open(out_csv + '.tmp', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(_FIELDS); w.writerows(rows)
    os.replace(out_csv + '.tmp', out_csv)
    if F is not TRAIN_FIELDS:
        del F
    gc.collect()
    print(f'{tag}: {hours/24:+7.2f} days, done in {time.time() - t0:.0f} s', flush=True)

# ------------------------------------------------------------
# Summary: time of every snapshot, tables and figure
# ------------------------------------------------------------
_all = []
for (path, t) in TIME_SOURCES:
    tag = f'{os.path.splitext(os.path.basename(path))[0]}_t{t:02d}'
    f = os.path.join(TEMPORAL_DIR, f'r2_{tag}.csv')
    if os.path.exists(f):
        _all += list(csv.DictReader(open(f)))
with open(os.path.join(TEMPORAL_DIR, 'temporal_r2_all.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=_FIELDS); w.writeheader(); w.writerows(_all)

_snaps = sorted({(r['file'], int(r['t_idx']), float(r['hours'])) for r in _all}, key=lambda s: s[2])
_tinfo = {(r['file'], int(r['t_idx'])): (float(r['days_since_init']), r['date']) for r in _all}
with open(os.path.join(TEMPORAL_DIR, 'temporal_snapshots.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['file', 't_idx', 'days_since_init', 'date', 'days_from_training'])
    for (fn, t, h) in _snaps:
        w.writerow([fn, t, f'{_tinfo[(fn, t)][0]:.4f}', _tinfo[(fn, t)][1], f'{h/24:.4f}'])

_R = {(r['file'], int(r['t_idx']), r['region'], r['model'], int(r['k'])): float(r['r2']) for r in _all}
for reg in REGIONS:
    print(f'\nR² on {reg} (rows: snapshots, by days since the training snapshot)')
    hdr = (f'{"file":>18} {"init+d":>8} {"date":>16} {"days":>8} '
           + ' '.join(f'{"LR k="+str(k):>9} {"FiLM k="+str(k):>10}' for k in K_PRED_cnn))
    print(hdr); print('-' * len(hdr))
    for (fn, t, h) in _snaps:
        print(f'{fn:>18} {_tinfo[(fn, t)][0]:>8.2f} {_tinfo[(fn, t)][1]:>16} {h/24:>8.2f} '
              + ' '.join(f'{_R[(fn, t, reg, "LR", k)]:>9.3f} {_R[(fn, t, reg, "FiLM", k)]:>10.3f}'
                         for k in K_PRED_cnn))

C_LR, C_FILM = '#C2410C', '#1D4ED8'
fig, axs = plt.subplots(len(REGIONS), len(K_PRED_cnn), figsize=(4.2 * len(K_PRED_cnn), 3.6 * len(REGIONS)),
                        sharex=True, squeeze=False, constrained_layout=True)
_d = [s[2] / 24 for s in _snaps]
for a, reg in enumerate(REGIONS):
    for b, k in enumerate(K_PRED_cnn):
        ax = axs[a, b]
        for m, c, mk in (('LR', C_LR, 's'), ('FiLM', C_FILM, 'o')):
            ax.plot(_d, [_R[(fn, t, reg, m, k)] for (fn, t, h) in _snaps], color=c, marker=mk, ms=4, lw=1.5, label=m)
        ax.axvline(0, color='0.6', lw=1, ls=':')
        ax.set_title(f'{reg}, k={k}, z={Z_LEVELS[k]:.1f} m'); ax.grid(alpha=0.3)
        if a == len(REGIONS) - 1: ax.set_xlabel('days since training snapshot')
        if b == 0: ax.set_ylabel(r'$R^2$')
axs[0, 0].legend(frameon=False)
plt.savefig(os.path.join(TEMPORAL_DIR, 'temporal_r2.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(TEMPORAL_DIR, 'temporal_r2.pdf'), bbox_inches='tight')
plt.close()
print(f'\nSaved in {TEMPORAL_DIR}: r2_<file>_tXX.csv, temporal_r2_all.csv, '
      f'temporal_snapshots.csv (time of every snapshot), temporal_r2.png/pdf')
