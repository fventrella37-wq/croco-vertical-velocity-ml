# AGU version of Figure 8 (fig8_ag.py): all text 8 pt at printed size (4.0 in wide).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ED=os.path.join(WORKDIR, 'cnn_seeds_film')
H=[pd.read_csv(f).set_index('epoch') for f in sorted(glob.glob(ED+'/seed_*/history.csv'))]
N=len(H)
common=sorted(set.intersection(*[set(h.index) for h in H]))
ep=np.array(common)
def stat(c):
    A=np.array([h.loc[common,c].values for h in H]); return A.mean(0),A.std(0,ddof=1)
Z={2:-11.2,7:-37.1,12:-71.1,17:-116.6}; col={2:'tab:blue',7:'tab:green',12:'tab:orange',17:'tab:red'}
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import agu_style; agu_style.apply()
fig,ax=plt.subplots(figsize=(4.0,3.5))
m,s=stat('train_loss')
ax.plot(ep,m,'k-',lw=1.3,label='training loss'); ax.fill_between(ep,m-s,m+s,color='k',alpha=0.25,lw=0)
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss (standardized MSE)')
ax2=ax.twinx()
lines=[ax.lines[0]]
for k in [2,7,12,17]:
    m,s=stat(f'val_r2_ag_k{k}')
    l,=ax2.plot(ep,m,color=col[k],lw=1.2,label=fr'$R^2$, $k={k}$ ($z={Z[k]}$ m)')
    ax2.fill_between(ep,m-s,m+s,color=col[k],alpha=0.2,lw=0); lines.append(l)
ax2.set_ylim(-0.2,1); ax2.set_ylabel(r'Out-of-sample $R^2$')
ax.set_xlim(0,ep.max())
lines=[lines[i] for i in (0,2,4,1,3)]  # row-wise order: loss,k2,k7 / k12,k17
fig.legend(lines,[l.get_label() for l in lines],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,0.0))
ax.set_title(f'Transition region ({N} seeds)')
fig.tight_layout(rect=(0,0.15,1,1))
d=FIGDIR
fig.savefig(d+'/train_loss_vs_val_r2_film_ensemble_algerian_agu.pdf')
fig.savefig(d+'/train_loss_vs_val_r2_film_ensemble_algerian_agu.png',dpi=300)
print('N',N,'last common epoch',ep.max())
