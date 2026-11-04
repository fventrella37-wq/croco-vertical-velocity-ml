"""Input and output folders used by the scripts in this folder.

WORKDIR : folder with the CROCO files (croco_avg.*.nc, grd_alb500sm.nc), the notebooks
          and the result folders (cnn_seeds_film/, lr_perlevel_run_*/, temporal_validation*/).
          Default: the repository root.
FIGDIR  : where the figures are written. Default: <repository root>/figures_out.
CACHE   : intermediate files (prep.npz, prep2.npz, pred.npz). Default: <repository root>/cache.
Each one can be changed with the environment variables VV_WORKDIR, VV_FIGDIR, VV_CACHE.
"""
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.environ.get('VV_WORKDIR', _ROOT)
FIGDIR = os.environ.get('VV_FIGDIR', os.path.join(_ROOT, 'figures_out'))
CACHE = os.environ.get('VV_CACHE', os.path.join(_ROOT, 'cache'))
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
