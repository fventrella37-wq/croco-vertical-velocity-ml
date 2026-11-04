# AGU version of the notebook figures: runs the same notebook cells as runcells.py, with
# all text at 8 pt as printed (common size of all the AGU figures) and the figure size scaled by Q so
# that the printed width equals the width at which the figure is included in the paper.
# Usage: python3 runcells_agu.py <corr|matrix|lrcoef|lrmaps> <Q>
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import json, os, sys, re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.figure
from matplotlib.colors import SymLogNorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib, xarray as xr
LAB, TCK = 8.0, 8.0
which, Q = sys.argv[1], float(sys.argv[2])
OUT = FIGDIR
TARGET = {'corr': ['w_correlation_vs_depth'], 'matrix': ['w_correlation_matrix_depth'],
          'lrcoef': ['lr_perlevel_coefficients_v2'],
          'lrmaps': ['val_alboran_C_LR_perlevel', 'val_algerian_C_LR_perlevel']}[which]
plt.show = lambda *a, **k: plt.close('all')
_orig = matplotlib.figure.Figure.savefig
def _save(self, fname, *a, **k):
    stem, ext = os.path.splitext(os.path.basename(str(fname)))
    if stem in TARGET and ext in ('.eps', '.pdf'):
        out = os.path.join(OUT, stem + '_agu.pdf')
        k.pop('dpi', None)
        _orig(self, out, bbox_inches='tight', pad_inches=0.02, **{kk: vv for kk, vv in k.items() if kk not in ('bbox_inches', 'pad_inches')})
        bb = self.get_tightbbox(self.canvas.get_renderer())
        print(f'SAVED {out}  width {bb.width:.3f} in  height {bb.height:.3f} in', flush=True)
matplotlib.figure.Figure.savefig = _save
plt.savefig = lambda fname, *a, **k: plt.gcf().savefig(fname, *a, **k)
SIZE = {'font.size': TCK, 'axes.titlesize': LAB, 'axes.labelsize': LAB, 'figure.titlesize': LAB,
        'xtick.labelsize': TCK, 'ytick.labelsize': TCK, 'legend.fontsize': TCK, 'legend.title_fontsize': TCK}
def fix(src):
    src = re.sub(r"'(font\.size|axes\.titlesize|axes\.labelsize|figure\.titlesize|xtick\.labelsize|ytick\.labelsize|legend\.fontsize|legend\.title_fontsize)'\s*:\s*[0-9.]+",
                 lambda m: f"'{m.group(1)}': {SIZE[m.group(1)]}", src)
    src = re.sub(r"fontsize\s*=\s*[0-9.]+", f"fontsize={TCK}", src)
    # line widths scaled with the figure, so that they print as thick as before
    src = re.sub(r"\b(lw|linewidth)\s*=\s*([0-9.]+)", lambda m: f"{m.group(1)}={float(m.group(2)) * Q:.3g}", src)
    src = re.sub(r"figsize\s*=\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
                 lambda m: "figsize=tuple(" + str(Q) + "*_v for _v in (" + m.group(1) + "))", src)
    return src
NB = os.path.join(WORKDIR, 'notebook_nuredduna.ipynb'); nb = json.load(open(NB))
def cell(marker):
    idx = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
    assert len(idx) == 1, marker
    return ''.join(nb['cells'][idx[0]]['source'])
d = np.load(os.path.join(CACHE, 'prep2.npz'))
ZL = d['Z_LEVELS'][:32].copy()
G = dict(np=np, plt=plt, os=os, SymLogNorm=SymLogNorm, LinearRegression=LinearRegression, r2_score=r2_score,
         joblib=joblib, xr=xr)
for nm in ['IB', 'NA', 'AG']:
    W = d[nm + '_WA']; U = np.full(W.shape, np.nan); V = np.full(W.shape, np.nan); U[2] = d[nm + '_U']; V[2] = d[nm + '_V']
    G['U_' + nm] = U; G['V_' + nm] = V; G['W_' + nm] = W; G['lat_' + nm] = d[nm + '_lat']; G['lon_' + nm] = d[nm + '_lon']
G['Z_LEVELS'] = ZL
wd = os.path.join(CACHE, 'run_agu_' + which); os.makedirs(wd, exist_ok=True); os.chdir(wd)
HB = ("HBL_FALLBACK = {'Alboran': (9.0, 32.0, 56.0), 'Transition': (25.0, 47.0, 68.0),\n                'SW Med (training)': (36.0, 66.0, 85.0)}",
      "HBL_FALLBACK = {'Alboran': (8.9, 32.2, 56.0), 'Transition': (24.5, 46.8, 67.9),\n                'SW Med (training)': (36.2, 66.4, 85.2)}")
if which in ('corr', 'matrix'):
    s = cell("def corr_with_ref_level(").replace(*HB); assert HB[1] in s
    exec(fix(s), G)
    if which == 'matrix':
        s = cell("def vertical_corr_matrix(").replace(*HB); assert HB[1] in s
        # AGU version: colour-bar label on two lines (the panels are now short), no seams between cells
        s = s.replace("label='Correlation of w between two levels'", "label='Correlation of w\\nbetween two levels'")
        s = s.replace("vmin=-1, vmax=1, shading='flat')", "vmin=-1, vmax=1, shading='flat', edgecolors='face', linewidth=0)")
        # fewer depth ticks and a one-column legend, so that the 8 pt labels do not overlap at print size
        s = s.replace("(10, 25, 50, 75, 100, 150, 200, 300, 400)", "(10, 50, 100, 200, 300)")
        s = s.replace("ncol=2, frameon=False, handlelength=3.2", "ncol=1, frameon=False, handlelength=3.2")
        # larger panels: taller figure (height factor MAT_H, env), depth labels only on the first panel
        s = s.replace("figsize=(5.2 * len(R_by_zone), 5.6)", "figsize=(5.2 * len(R_by_zone), " + os.environ.get('MAT_H', '5.6') + ")")
        s = s.replace("between two levels', shrink=0.9)", "between two levels', shrink=0.9, ticks=[-1, -0.5, 0, 0.5, 1])")
        s = s.replace("    axs[0, 0].set_ylabel('Depth (m)')", "    axs[0, 0].set_ylabel('Depth (m)')\n    for _a in axs[0, 1:]: _a.tick_params(labelleft=False)")
        assert '_a.tick_params(labelleft=False)' in s
        assert 'between two levels' in s and "edgecolors='face'" in s and '(10, 50, 100, 200, 300)' in s and 'ncol=1, frameon' in s
        exec(fix(s), G)
else:
    s1 = cell("val_algerian_C_LR_perlevel.png"); s2 = cell("lr_validation_r2_all_levels.csv"); s3 = cell("K_INSPECT_C = [4, 10]")
    exec(fix(s1) if which == 'lrmaps' else s1, G)
    if which == 'lrcoef':
        # AGU version: axis labels only on the outer panels (they are the same on all panels),
        # one colour bar per row (the two panels of a row share the colour scale), no title
        R3 = [("""        ax.set_xlabel(r'$\\Delta i$ ($\\xi$ axis, native grid)')
        ax.set_ylabel(r'$\\Delta j$ ($\\eta$ axis, native grid)')""",
               """        if row == len(K_INSPECT_C) - 1: ax.set_xlabel(r'$\\Delta i$ ($\\xi$ axis, native grid)')
        else: ax.tick_params(labelbottom=False)
        if col == 0: ax.set_ylabel(r'$\\Delta j$ ($\\eta$ axis, native grid)')
        else: ax.tick_params(labelleft=False)"""),
              ("        ax.set_title(f'{label}, k={k_abs}, z={Z_LEVELS[k_abs]:.1f} m')",
               "        ax.set_title(f'${label}$, $k={k_abs}$, $z={Z_LEVELS[k_abs]:.1f}$ m')"),
              ("        fig.colorbar(im, ax=ax, label='weight')",
               "        if col == 1: fig.colorbar(im, ax=axs[row, :], label='Weight', shrink=0.9)"),
              ("figsize=(7.5, 3.6*len(K_INSPECT_C))", "figsize=(7.5, 3.15*len(K_INSPECT_C))"),
              ("plt.suptitle('Coefficient maps, native grid coordinates (\u03be, \u03b7)')\n", "")]
        for _a, _b in R3:
            assert s3.count(_a) == 1, _a
            s3 = s3.replace(_a, _b)
        exec(s2, G); exec(fix(s3), G)
print('done', which)
