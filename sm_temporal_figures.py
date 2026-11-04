# Figures of the Supplementary Material on the temporal validation (print size,
# AGU text width 5.5 in, all text 8 pt), saved as vector PDF in Draft/figures/sm_temporal_*.pdf.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import os, json, numpy as np, pandas as pd, netCDF4
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Rectangle
C = WORKDIR; T = os.path.join(C, 'temporal_validation')
SUM = os.path.join(C, 'temporal_validation_summer_2136'); OUT = FIGDIR
# all text 8 pt as printed (common style of the AGU figures), at the AGU text width of 5.5 in
LAB, TCK = 8.0, 8.0
plt.rcParams.update({'font.size': TCK, 'axes.titlesize': LAB, 'axes.labelsize': LAB, 'xtick.labelsize': TCK,
                     'ytick.labelsize': TCK, 'legend.fontsize': TCK, 'mathtext.fontset': 'cm',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'lines.linewidth': 0.9, 'lines.markersize': 2.4, 'pdf.fonttype': 42})
W_IN = 5.5                                   # AGU text width
REG = ['Alboran', 'Transition']; RT = {'Alboran': 'Alboran', 'Transition': 'Transition', 'Train': 'SW Med (training)'}
K = [2, 7, 12, 17]
C_LR, C_FILM = '#C2410C', '#1D4ED8'
T_WIN, T_SUM = pd.Timestamp('2011-02-03 19:30'), pd.Timestamp('2011-08-07 19:30')
P0, P1 = pd.Timestamp('2011-05-29 19:30'), pd.Timestamp('2011-10-23 19:30')   # summer period (22 snapshots)

# z of the fixed levels (same as the notebooks)
with netCDF4.Dataset(f'{C}/grd_alb500sm.nc') as g: h = np.asarray(g['h'][1:-1, 1:-1], float)
with netCDF4.Dataset(f'{C}/croco_avg.0656.nc') as d:
    hc = float(d.hc); sc = np.asarray(d.sc_r, float); cs = np.asarray(d.Cs_r, float)
    j, i = np.unravel_index(np.nanargmax(h), h.shape); z0 = float(d['zeta'][0, j + 1, i + 1])
Hd = h[j, i]; Z = np.sort(z0 + (z0 + Hd) * (hc * sc + Hd * cs) / (hc + Hd))[::-1]

E = pd.read_csv(os.path.join(SUM, 'eval_all.csv'), parse_dates=['date']).sort_values('date')
H = pd.read_csv(os.path.join(T, 'temporal_hbl.csv'), parse_dates=['date']).sort_values('date')
Wc = pd.read_csv(os.path.join(T, 'temporal_wcorr.csv'), parse_dates=['date']).sort_values('date')

def date_axis(ax, mark=(), months=(1, 7)):
    ax.axvspan(P0 - pd.Timedelta(days=3.5), P1 + pd.Timedelta(days=3.5), color='0.92', lw=0, zorder=0)
    for t in mark:
        ax.axvline(t, color='k', ls=':', lw=0.9, zorder=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=list(months)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.set_xlim(pd.Timestamp('2010-12-25'), pd.Timestamp('2012-02-20'))
    ax.grid(alpha=0.3, lw=0.4)

def grid_2x4(col, tr, ylab, hline, ylims=None, mark=()):
    fig, axs = plt.subplots(2, 4, figsize=(W_IN, 3.5), sharex=True)
    lims = {}
    for a, reg in enumerate(REG):
        for b, k in enumerate(K):
            ax = axs[a, b]
            for m, c, mk in (('LR', C_LR, 's'), ('FiLM', C_FILM, 'o')):
                s = E[(E.region == reg) & (E.training == tr) & (E.model == m) & (E.k == k)]
                ax.plot(s.date, s[col], color=c, marker=mk, label=m, zorder=3)
            ax.axhline(hline, color='0.4', lw=0.6, zorder=2)
            date_axis(ax, mark)
            if a == 0: ax.set_title(f'$k={k}$, $z={Z[k]:.1f}$ m', pad=3)
            if ylims: ax.set_ylim(*ylims[(reg, k)])
            lims[(reg, k)] = ax.get_ylim()
            if b == 0: ax.set_ylabel(f'{RT[reg]}\n{ylab}')
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    h_ += [plt.Line2D([], [], color='k', ls=':', lw=0.9), Rectangle((0, 0), 1, 1, color='0.92')]
    l_ += ['training snapshot', 'summer period (29 May to 23 Oct 2011)']
    fig.legend(h_, l_, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=0.6, w_pad=0.4)
    return fig, lims

# common y range for R² of winter- and summer-trained models
yl = {}
for reg in REG:
    for k in K:
        s = E[(E.region == reg) & (E.k == k)].r2
        lo, hi = s.min(), s.max(); pad = 0.05 * (hi - lo)
        yl[(reg, k)] = (min(lo - pad, -0.05), max(hi + pad, 0.05))
fig, _ = grid_2x4('r2', 'winter', r'$R^2$', 0.0, yl, (T_WIN,)); fig.savefig(f'{OUT}/sm_temporal_r2_winter.pdf'); plt.close(fig)
fig, _ = grid_2x4('r2', 'summer', r'$R^2$', 0.0, yl, (T_SUM,)); fig.savefig(f'{OUT}/sm_temporal_r2_summer.pdf'); plt.close(fig)
fig, _ = grid_2x4('std_ratio', 'winter', r'$\sigma_{\hat w}\,/\,\sigma_w$', 1.0, None, (T_WIN,)); fig.savefig(f'{OUT}/sm_temporal_amplitude_winter.pdf'); plt.close(fig)

# boundary layer and stratification
fig, axs = plt.subplots(2, 2, figsize=(W_IN, 3.5), sharex=True)
for jx, reg in enumerate(REG):
    s = H[H.region == reg]
    ax = axs[0, jx]
    ax.fill_between(s.date, s.hbl_p10, s.hbl_p90, color='0.75', lw=0, label=r'$10$th to $90$th percentile', zorder=2)
    ax.plot(s.date, s.hbl_p50, 'k-o', label='median', zorder=3)
    for k in K:
        ax.axhline(-Z[k], color='tab:blue', lw=0.6, ls='--', zorder=2)
    ax.set_ylim(125, 0); ax.set_ylabel(r'$h_{\rm bl}$ (m)'); ax.set_title(RT[reg], pad=3)
    date_axis(ax, (T_WIN,), (1, 4, 7, 10))
    ax2 = axs[1, jx]
    ax2.plot(s.date, s.dT_med, 'k-o', zorder=3)
    ax2.set_ylabel(r'$T(-11.2\,{\rm m})-T(-116.6\,{\rm m})$ ($^\circ$C)')
    date_axis(ax2, (T_WIN,), (1, 4, 7, 10))
h_, l_ = axs[0, 0].get_legend_handles_labels()
h_ += [plt.Line2D([], [], color='tab:blue', ls='--', lw=0.6)]; l_ += [r'target levels $k=2,7,12,17$']
fig.legend(h_, l_, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6); fig.savefig(f'{OUT}/sm_temporal_hbl.pdf'); plt.close(fig)

# depth-time map of the vertical correlation
rc = [c for c in Wc.columns if c.startswith('r_k')]; ks = [int(c[3:]) for c in rc]; depth = -Z[ks]
fig, axs = plt.subplots(1, 2, figsize=(W_IN, 2.5), sharey=True)
for jx, reg in enumerate(REG):
    s, hh = Wc[Wc.region == reg], H[H.region == reg]
    ax = axs[jx]
    pc = ax.pcolormesh(s.date, depth, s[rc].values.T, cmap='RdBu_r', vmin=-1, vmax=1, shading='nearest', lw=0, edgecolor='face')
    ax.plot(hh.date, hh.hbl_p50, 'k-', lw=1.0, label=r'$h_{\rm bl}$ median')
    ax.plot(hh.date, hh.hbl_p90, 'k--', lw=1.0, label=r'$h_{\rm bl}$ $90$th percentile')
    ax.axvline(T_WIN, color='k', ls=':', lw=0.9)
    ax.set_ylim(400, 0); ax.set_title(RT[reg], pad=3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
axs[0].set_ylabel('Depth (m)')
axs[0].legend(loc='lower left', framealpha=0.9)
fig.tight_layout(rect=(0, 0, 0.855, 1))
cax = fig.add_axes([0.87, 0.2, 0.014, 0.68]); fig.colorbar(pc, cax=cax, label=r'$r\,[w(-11.2\,{\rm m}),\,w(z)]$')
fig.savefig(f'{OUT}/sm_temporal_wcorr.pdf'); plt.close(fig)

# vertical correlation profiles, 3 Feb and 7 Aug 2011, three domains
DD = os.path.join(T, 'feb_aug_with_training')
W2 = pd.read_csv(os.path.join(DD, 'temporal_wcorr.csv')); H2 = pd.read_csv(os.path.join(DD, 'temporal_hbl.csv'))
ST = {'Alboran': ('#c0392b', '#e6a39b'), 'Transition': ('#8e44ad', '#cfa8e0'), 'Train': ('#1e8449', '#9fd0b1')}
fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.4), sharey=True)
for ax, (dt, title) in zip(axs, [('2011-02-03 19:30', '3 February 2011'), ('2011-08-07 19:30', '7 August 2011')]):
    ax.axhspan(-Z[2], -Z[17], color='#e6f2e6', zorder=0, label=r'target levels $k=2$ to $17$')
    ax.axhline(-Z[2], color='gray', ls='--', lw=0.8, label=r'stencil level $k=2$ ($-11.2$ m)')
    for n, (reg, (c, cl)) in enumerate(ST.items()):
        r = W2[(W2.region == reg) & (W2.date == dt)][rc].values[0]
        ax.plot(r, depth, '-', color=c, lw=1.2, label=RT[reg], zorder=2)
        hb = H2[(H2.region == reg) & (H2.date == dt)].iloc[0]
        x0 = -0.17 + 0.05 * n
        ax.add_patch(Rectangle((x0, hb.hbl_p10), 0.035, max(hb.hbl_p90 - hb.hbl_p10, 0.8),
                               facecolor=cl, edgecolor=c, lw=0.7, zorder=3))
        ax.plot([x0 - 0.006, x0 + 0.041], [hb.hbl_p50, hb.hbl_p50], color=c, lw=1.6, zorder=4, solid_capstyle='butt')
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_ylim(400, 0); ax.set_xlim(-0.2, 1.02); ax.grid(True, color='0.9', lw=0.4)
    ax.set_xlabel(r'$r\,[w(-11.2\,{\rm m}),\,w(z)]$'); ax.set_title(title, pad=3)
axs[0].set_ylabel('Depth (m)')
h_, l_ = axs[0].get_legend_handles_labels()
h_.append(Rectangle((0, 0), 0, 0, facecolor='0.8', edgecolor='0.3', lw=0.7)); l_.append(r'$h_{\rm bl}$: $10$th to $90$th percentile (bar), median (tick)')
fig.legend(h_, l_, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5)
fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.savefig(f'{OUT}/sm_temporal_wcorr_profiles.pdf'); plt.close(fig)
print('done')
