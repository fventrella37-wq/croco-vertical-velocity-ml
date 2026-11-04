# Reproduce the notebook preprocessing (xcroco) for the three domains with h5py + numpy,
# only for what Fig. 10 needs. Output: npz with U,V at k=2 and W at levels 2..17 per domain.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import h5py, numpy as np, os
D=WORKDIR
f=h5py.File(f'{D}/croco_avg.0656.nc','r'); g=h5py.File(f'{D}/grd_alb500sm.nc','r')
A=f.attrs
hc=float(np.ravel(A['hc'])[0]); sc_r=np.ravel(A['sc_r']).astype(float); Cs_r=np.ravel(A['Cs_r']).astype(float)
sc_w=np.ravel(A['sc_w']).astype(float); Cs_w=np.ravel(A['Cs_w']).astype(float)
T=0
lat=g['lat_rho'][1:-1,1:-1]; lon=g['lon_rho'][1:-1,1:-1]; h=g['h'][1:-1,1:-1]
zeta_full=f['zeta'][T,1:-1,1:-1].astype(float)
def zcoord(hh,zz,sc,cs):
    z0=(hc*sc[:,None,None]+np.abs(hh)[None]*cs[:,None,None])/(hc+np.abs(hh)[None])
    return z0*(zz+hh)[None]+zz[None]
jd,idd=np.unravel_index(np.nanargmax(h),h.shape)
Z=np.sort(zcoord(h[jd:jd+1,idd:idd+1],zeta_full[jd:jd+1,idd:idd+1],sc_r,Cs_r)[:,0,0])  # bottom->surface
ZL=Z[::-1]   # k=0 surface
print('deep',jd,idd,'Z_LEVELS k=2..17:',np.round(ZL[2:18],2))
KS=list(range(2,18))
def interp_col(src_z, src_v, targets):
    # linear interpolation along axis 0 for every column; NaN outside [min,max]
    nz,ny,nx=src_z.shape; out=np.full((len(targets),ny,nx),np.nan)
    for t,zt in enumerate(targets):
        # index of first level with z >= zt (z increasing with index)
        k=(src_z<zt).sum(0)
        ok=(k>0)&(k<nz)
        kk=np.clip(k,1,nz-1)
        z0=np.take_along_axis(src_z,(kk-1)[None],0)[0]; z1=np.take_along_axis(src_z,kk[None],0)[0]
        v0=np.take_along_axis(src_v,(kk-1)[None],0)[0]; v1=np.take_along_axis(src_v,kk[None],0)[0]
        r=(zt-z0)/(z1-z0); o=v0+r*(v1-v0); o[~ok]=np.nan
        # exact hit on the top level
        out[t]=o
    return out
boxes={'IB':(0.40,2.00,36.90,38.30),'NA':(-3.60,-2.20,35.70,36.40),'AG':(-1.50,-0.20,36.40,37.40)}
res={'Z_LEVELS':ZL}
for nm,(a,b,c,d) in boxes.items():
    m=(lat>=c)&(lat<=d)&(lon>=a)&(lon<=b)
    jj,ii=np.where(m); j0,j1,i0,i1=jj.min(),jj.max()+1,ii.min(),ii.max()+1
    # file rho index = inner + 1
    J0,J1,I0,I1=j0+1,j1+1,i0+1,i1+1
    uf=f['u'][T,:,J0:J1,I0-1:I1].astype(float)        # u_file[i-1], u_file[i] around rho file i
    vf=f['v'][T,:,J0-1:J1,I0:I1].astype(float)
    u=0.5*(uf[:,:,:-1]+uf[:,:,1:]); v=0.5*(vf[:,:-1,:]+vf[:,1:,:])
    w=f['w'][T,:,J0:J1,I0:I1].astype(float)
    hh=h[j0:j1,i0:i1]; zz=zeta_full[j0:j1,i0:i1]
    zr=zcoord(hh,zz,sc_r,Cs_r); zw=zcoord(hh,zz,sc_w,Cs_w)
    U=interp_col(zr,u,[ZL[2]])[0]; V=interp_col(zr,v,[ZL[2]])[0]
    W=interp_col(zw,w,ZL[KS])
    mk=~m[j0:j1,i0:i1]
    U[mk]=np.nan; V[mk]=np.nan; W[:,mk]=np.nan
    res[nm+'_U']=U; res[nm+'_V']=V; res[nm+'_W']=W
    res[nm+'_lat']=lat[j0:j1,i0:i1]; res[nm+'_lon']=lon[j0:j1,i0:i1]
    print(nm,U.shape,'valid',np.isfinite(U).sum())
np.savez_compressed(os.path.join(CACHE, 'prep.npz'),**res)
print('saved')
