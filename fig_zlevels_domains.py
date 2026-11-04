# Figure 1 of the AGU version: (a) fixed z-levels, (b) the three domains on the CROCO
# bathymetry (same data, boxes and colours as the notebook cell of domains_map_v2).
# All text 8 pt at printed size (full text width). The bathymetry field of panel (b)
# is drawn as an image at 600 dpi (as the former domains_map_v2.png); text and lines vector.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import os, sys, numpy as np, netCDF4
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import agu_style; agu_style.apply()
C = WORKDIR; F = FIGDIR
with netCDF4.Dataset(f'{C}/grd_alb500sm.nc') as g:
    lon_g = np.asarray(g['lon_rho'][:]); lat_g = np.asarray(g['lat_rho'][:])
    h_g = np.asarray(g['h'][:], float); mask_g = np.asarray(g['mask_rho'][:])
h = h_g[1:-1, 1:-1]
with netCDF4.Dataset(f'{C}/croco_avg.0656.nc') as d:
    hc = float(d.hc); sc = np.asarray(d.sc_r, float); cs = np.asarray(d.Cs_r, float)
    j, i = np.unravel_index(np.nanargmax(h), h.shape); z0 = float(d['zeta'][0, j + 1, i + 1])
Hd = h[j, i]; Z = np.sort(z0 + (z0 + Hd) * (hc * sc + Hd * cs) / (hc + Hd))[::-1]
print(len(Z), round(Z[0], 2), round(Z[-1], 1))
h_ocean = np.where(mask_g > 0, h_g, np.nan)
DOMAINS = [('South-West Mediterranean (training)', (0.40, 2.00), (36.90, 38.30), 'lime'),
           ('Alboran (validation)', (-3.60, -2.20), (35.70, 36.40), 'red'),
           ('Transition region (validation)', (-1.50, -0.20), (36.40, 37.40), 'magenta')]
W = agu_style.TEXTWIDTH_IN; H = 2.75
fig = plt.figure(figsize=(W, H))
axa = fig.add_axes([0.135, 0.30, 0.12, 0.64])
for z in Z:
    axa.axhline(z, color='tab:blue', lw=0.45)
axa.set_ylim(-3050, 60); axa.set_xticks([]); axa.set_ylabel('Depth (m)')
axb = fig.add_axes([0.385, 0.30, 0.485, 0.64])
pc = axb.pcolormesh(lon_g, lat_g, h_ocean, shading='auto', cmap='Blues_r', rasterized=True)
for label, (lo0, lo1), (la0, la1), col in DOMAINS:
    axb.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, edgecolor=col, lw=1.4, label=label))
axb.set_xlabel('Longitude'); axb.set_ylabel('Latitude')
fig.legend(*axb.get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, handlelength=1.4,
           columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
cax = fig.add_axes([0.885, 0.30, 0.015, 0.64]); fig.colorbar(pc, cax=cax, label='Depth (m)')
fig.text(0.005, 0.955, '(a)'); fig.text(0.305, 0.955, '(b)')
fig.savefig(f'{F}/zlevels_domains.pdf', dpi=600)
fig.savefig(f'{F}/zlevels_domains.png', dpi=300)
print('saved')
