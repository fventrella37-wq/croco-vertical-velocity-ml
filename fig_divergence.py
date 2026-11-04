# Figure 3 of the AGU version: distribution of the 3D divergence on the training
# domain (same data and histogram as notebook_nuredduna.ipynb, cell 24), full text width,
# without title, with legend and statistics outside the axes.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
C = WORKDIR; F = FIGDIR
d = np.load(f'{C}/derivate_oceaniche.npz')
div = d['duo_dx'] + d['dvo_dy'] + d['dwo_dz']
r = div[np.isfinite(div)]
mean_r, std_r = float(r.mean()), float(r.std())
p0, p100 = np.percentile(r, [1, 99])
counts, edges = np.histogram(r, bins=100000, range=(p0, p100))
S = 1e5                                   # x axis in units of 1e-5 s^-1
# probability density with respect to the plotted variable (units of 1e-5 s^-1),
# normalized with the total number of points, so that its integral over the full range is 1
pdf = counts / r.size / ((edges[1] - edges[0]) * S)
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import agu_style; agu_style.apply()
# all text 8 pt; drawn at printed size (included at natural size)
fig = plt.figure(figsize=(5.5, 2.4))            # full text width
ax = fig.add_axes([0.075, 0.175, 0.645, 0.79])
ax.fill_between(edges[:-1] * S, pdf, step='post', color='steelblue', lw=0, label='CROCO')
x = np.linspace(p0, p100, 1000)
ax.plot(x * S, norm.pdf(x * S, mean_r * S, std_r * S), color='red', lw=1.0, label='Gaussian fit')
ax.axvline(0, color='black', lw=1.0, label='zero')
ax.axvline(mean_r * S, color='orange', lw=1.0, ls='--', label='mean')
ax.set_xlim(p0 * S, p100 * S); ax.set_yscale('log')
ax.set_xlabel(r'$\nabla_{\xi\eta z}\cdot\mathbf{u}$ ($10^{-5}$ s$^{-1}$)')
leg = ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0), frameon=False, borderaxespad=0)
fig.text(0.745, 0.42, f'mean $= {mean_r / 1e-8:.2f}\\times10^{{-8}}$ s$^{{-1}}$\nstd $= {std_r / 1e-6:.2f}\\times10^{{-6}}$ s$^{{-1}}$',
         va='top', ha='left')
fig.savefig(f'{F}/divergence_distribution_3D_croco_agu.pdf')
fig.savefig(f'{F}/divergence_distribution_3D_croco_agu.png', dpi=300)
print('%.3g %.3g' % (mean_r, std_r), 'saved')
