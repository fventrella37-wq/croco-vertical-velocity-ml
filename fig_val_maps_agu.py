# Validation maps of the AGU manuscript (Figures 6 and 9): predicted and CROCO w at the
# four target levels k = 2, 7, 12, 17, on (a) Alboran and (b) the transition region, in one
# figure at the AGU text width (5.5 in), all text 8 pt as printed.
# Usage: python3 fig_val_maps_agu.py <LR|FiLM>
#   LR   : per-level linear regression (7x7 stencil, 98 features), refitted here on the
#          SW Med training domain exactly as in the notebook, R2 checked against Table 1.
#   FiLM : FiLM CNN predictions of seed 261 (CACHE/pred.npz).
# The maps are vector graphics: the grid cells of the same colour (256 colours of the
# colour map) are drawn as one compound path, coordinates written to 0.01 pt.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from matplotlib.colors import SymLogNorm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import _path
import matplotlib.backends.backend_pdf as bp
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agu_style; agu_style.apply()

def _ops(path, transform, clip=None, simplify=None, sketch=None):
    return [bp.Verbatim(_path.convert_to_string(path, transform, clip, simplify, sketch, 2,
            [bp.Op.moveto.value, bp.Op.lineto.value, b'', bp.Op.curveto.value, bp.Op.closepath.value], True))]
bp.PdfFile.pathOperations = staticmethod(_ops)

which = sys.argv[1]
d = np.load(os.path.join(CACHE, 'prep.npz')); ZL = d['Z_LEVELS']
OUT = FIGDIR; K = [2, 7, 12, 17]; R = 3
DOM = [('NA', '(a) Alboran'), ('AG', '(b) Transition region')]

def stencils(U, V):
    su, sv = swv(U, (2 * R + 1, 2 * R + 1)), swv(V, (2 * R + 1, 2 * R + 1))
    X = np.concatenate([su.reshape(*su.shape[:2], -1), sv.reshape(*sv.shape[:2], -1)], axis=-1)
    return X, ~np.isnan(X).any(-1)

def inner(a): return a[R:-R, R:-R]

pred = {}
if which == 'LR':
    X, ok = stencils(d['IB_U'], d['IB_V'])
    models = {}
    for k in K:
        w = inner(d['IB_W'][k - 2]); m = ok & ~np.isnan(w)
        models[k] = LinearRegression().fit(X[m], w[m])
        print(f'in-sample k={k}: R2 {r2_score(w[m], models[k].predict(X[m])):.3f}  N {m.sum()}')
    for nm, _ in DOM:
        X, ok = stencils(d[nm + '_U'], d[nm + '_V'])
        for k in K:
            w = d[nm + '_W'][k - 2]; m = ok & ~np.isnan(inner(w))
            p = np.full(w.shape, np.nan); pi = inner(p); pi[m] = models[k].predict(X[m])
            pred[nm, k] = p
else:
    pp = np.load(os.path.join(CACHE, 'pred.npz'))
    for nm, _ in DOM:
        for k in K: pred[nm, k] = pp[f'{nm}_pred_{k}']
for nm, _ in DOM:
    for k in K:
        w, p = d[nm + '_W'][k - 2], pred[nm, k]; m = np.isfinite(w) & np.isfinite(p)
        print(f'{which} {nm} k={k}: R2 {r2_score(w[m], p[m]):.3f}  N {m.sum()}')

CMAP = plt.get_cmap('RdBu_r'); LUT = CMAP(np.arange(CMAP.N))
def draw(ax, lon, lat, data, norm):
    qm = ax.pcolormesh(lon, lat, np.zeros_like(lon), shading='nearest'); C = qm.get_coordinates(); qm.remove()
    j, i = np.nonzero(np.isfinite(data))
    idx = np.clip(np.floor(norm(data[j, i]).filled(np.nan) * CMAP.N), 0, CMAP.N - 1).astype(int)
    quad = np.stack([C[j, i], C[j, i + 1], C[j + 1, i + 1], C[j + 1, i], C[j, i]], axis=1)
    code = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY], np.uint8)
    for c in np.unique(idx):
        q = quad[idx == c]
        ax.add_patch(PathPatch(Path(q.reshape(-1, 2), np.tile(code, len(q))), facecolor=LUT[c],
                               edgecolor=LUT[c], lw=0.1, zorder=1))

def lonfmt(x, _): return f'{abs(x):.1f}°' + ('W' if x < 0 else 'E')
def latfmt(x, _): return f'{x:.1f}°N'

# layout (inches)
FW, L, RM, G = 5.5, 0.62, 0.04, 0.07
MW = (FW - L - RM - 3 * G) / 4
ext = {}
for nm, _ in DOM:
    v = np.isfinite(d[nm + '_W'][0]); lo, la = d[nm + '_lon'][v], d[nm + '_lat'][v]
    asp = 1 / np.cos(np.deg2rad(la.mean()))
    ext[nm] = (lo.min(), lo.max(), la.min(), la.max(), asp, MW * (la.max() - la.min()) * asp / (lo.max() - lo.min()))
TOP, HLAB, GAP, XT, CBG, CBH, CBT, BG = 0.20, 0.20, 0.05, 0.19, 0.03, 0.07, 0.26, 0.10
FH = TOP + sum(HLAB + 2 * ext[nm][5] + GAP + XT + CBG + CBH + CBT for nm, _ in DOM) + BG
fig = plt.figure(figsize=(FW, FH))
def ax_at(x, ytop, w, h): return fig.add_axes([x / FW, (FH - ytop - h) / FH, w / FW, h / FH])
y = TOP
for b, k in enumerate(K):
    fig.text((L + b * (MW + G) + MW / 2) / FW, 1 - 0.12 / FH, f'$k={k}$, $z={ZL[k]:.1f}$ m', ha='center', va='center')
for nm, title in DOM:
    lo0, lo1, la0, la1, asp, MH = ext[nm]
    fig.text(0.0, 1 - (y + HLAB / 2) / FH, title, ha='left', va='center')
    y += HLAB
    for a, (rowlab, src) in enumerate([('Predicted', 'p'), ('CROCO', 'w')]):
        yt = y + a * (MH + GAP)
        fig.text(0.06 / FW, 1 - (yt + MH / 2) / FH, rowlab, rotation=90, ha='center', va='center')
        for b, k in enumerate(K):
            w = d[nm + '_W'][k - 2]; vmax = np.nanmax(np.abs(w))
            norm = SymLogNorm(linthresh=vmax * 0.01, vmin=-vmax, vmax=vmax, base=10)
            ax = ax_at(L + b * (MW + G), yt, MW, MH)
            draw(ax, d[nm + '_lon'], d[nm + '_lat'], pred[nm, k] if src == 'p' else w, norm)
            ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1); ax.set_aspect(asp)
            ax.xaxis.set_major_locator(mt.FixedLocator([-3.0, -2.5] if nm == 'NA' else [-1.0, -0.5])); ax.yaxis.set_major_locator(mt.MultipleLocator(0.2 if nm == 'NA' else 0.3))
            ax.xaxis.set_major_formatter(mt.FuncFormatter(lonfmt)); ax.yaxis.set_major_formatter(mt.FuncFormatter(latfmt))
            ax.tick_params(length=2, pad=1.5)
            if a == 0: ax.tick_params(labelbottom=False)
            if b > 0: ax.tick_params(labelleft=False)
            if a == 1:
                cax = ax_at(L + b * (MW + G) + 0.08 * MW, yt + MH + XT + CBG, 0.84 * MW, CBH)
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal')
                e = int(np.floor(np.log10(vmax)))
                # labelled ticks at 0 and +-10^e, unlabelled ticks at +-10^(e-1) (labels would overlap)
                cb.set_ticks([-10.0**e, 0, 10.0**e]); cb.set_ticklabels([f'$-10^{{{e}}}$', '$0$', f'$10^{{{e}}}$'])
                cb.ax.xaxis.set_minor_locator(mt.FixedLocator([-10.0**(e - 1), 10.0**(e - 1)]))
                cb.ax.tick_params(length=2, pad=1.5); cb.ax.tick_params(which='minor', length=1.5); cb.outline.set_linewidth(0.5)
                if b == 0:
                    fig.text((L - 0.08) / FW, 1 - (yt + MH + XT + CBG + CBH / 2) / FH, r'$w$ (m s$^{-1}$)', ha='right', va='center')
    y += 2 * MH + GAP + XT + CBG + CBH + CBT
out = os.path.join(OUT, f'val_maps_{which}_agu.pdf')
fig.savefig(out); print('saved', out, f'{FW:.2f} x {FH:.2f} in', os.path.getsize(out) // 1024, 'kB')
