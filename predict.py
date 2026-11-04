import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORKDIR, FIGDIR, CACHE   # input/output folders, see paths.py
import zipfile, pickle, numpy as np, os, json, collections
from numpy.lib.stride_tricks import sliding_window_view
D=os.path.join(WORKDIR, 'cnn_seeds_film/seed_261')
# ---- load a torch state_dict without torch
def load_pt(path):
    z=zipfile.ZipFile(path); names=z.namelist(); root=names[0].split('/')[0]
    class Storage:
        def __init__(s,key,dtype): s.key=key; s.dtype=dtype
    def _rebuild_tensor_v2(storage, offset, size, stride, *a):
        buf=np.frombuffer(z.read(f'{root}/data/{storage.key}'),dtype=storage.dtype)
        n=int(np.prod(size)) if len(size) else 1
        arr=np.lib.stride_tricks.as_strided(buf[offset:], shape=tuple(size), strides=tuple(s*buf.itemsize for s in stride))
        return np.array(arr)
    DT={'FloatStorage':np.float32,'DoubleStorage':np.float64,'LongStorage':np.int64}
    class U(pickle.Unpickler):
        def find_class(s,mod,name):
            if name=='_rebuild_tensor_v2': return _rebuild_tensor_v2
            if name=='OrderedDict': return collections.OrderedDict
            if name in DT: return name
            return super().find_class(mod,name)
        def persistent_load(s,pid):
            _,stype,key,loc,numel=pid
            return Storage(key,DT[stype if isinstance(stype,str) else stype])
    return U(z.open(f'{root}/data.pkl')).load()
sd=load_pt(f'{D}/model_final.pt')
print({k:v.shape for k,v in sd.items()})
def conv2d(x,W,b,pad):
    if pad: x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))
    win=sliding_window_view(x,(3,3),axis=(2,3))          # B,C,H,W,3,3
    return np.einsum('bchwij,ocij->bohw',win,W,optimize=True)+b[None,:,None,None]
relu=lambda a: np.maximum(a,0)
def predict(P,zs):   # P (B,2,7,7) standardized, zs (B,) standardized
    h=relu(conv2d(P,sd['conv1.weight'],sd['conv1.bias'],1))
    gb=zs[:,None]*sd['film1.to_gamma_beta.weight'][:,0][None]+sd['film1.to_gamma_beta.bias'][None]
    g,b=gb[:,:8],gb[:,8:]
    h=g[:,:,None,None]*h+b[:,:,None,None]
    h=relu(conv2d(h,sd['conv2.weight'],sd['conv2.bias'],0))
    f=h.reshape(len(h),-1)
    f=relu(f@sd['fc.0.weight'].T+sd['fc.0.bias'])
    return (f@sd['fc.3.weight'].T+sd['fc.3.bias'])[:,0]
d=np.load(os.path.join(CACHE, 'prep.npz')); ZL=d['Z_LEVELS']; KS=list(range(2,18)); r=3
def patches(U,V,W):
    uw=sliding_window_view(U,(7,7)); vw=sliding_window_view(V,(7,7))
    ok=~np.isnan(uw).any((2,3)) & ~np.isnan(vw).any((2,3))
    jj,ii=np.where(ok)
    P=np.stack([uw[jj,ii],vw[jj,ii]],1); cj,ci=jj+r,ii+r
    return P, W[:,cj,ci].T, cj, ci
Pib,Yib,_,_=patches(d['IB_U'],d['IB_V'],d['IB_W'])
mu,su=Pib[:,0].mean(),Pib[:,0].std(); mv,sv=Pib[:,1].mean(),Pib[:,1].std()
zl=ZL[KS]; mz,sz=zl.mean(),zl.std()
mw=np.array([np.nanmean(Yib[:,t]) for t in range(16)]); sw=np.array([np.nanstd(Yib[:,t]) for t in range(16)])+1e-10
print('IB patches',len(Pib))
ref=json.load(open(f'{D}/result.json'))['r2_final']
out={}
for nm,key in [('NA','Alboran'),('AG','Algerian')]:
    P,Y,cj,ci=patches(d[nm+'_U'],d[nm+'_V'],d[nm+'_W'])
    Ps=P.astype(np.float32).copy(); Ps[:,0]=(P[:,0]-mu)/su; Ps[:,1]=(P[:,1]-mv)/sv
    print(nm,'patches',len(P))
    for k in [2,7,12,17]:
        t=k-2; idx=np.where(~np.isnan(Y[:,t]))[0]
        zs=np.full(len(idx),(ZL[k]-mz)/sz)
        yp=np.concatenate([predict(Ps[idx[s:s+4096]].astype(np.float64),zs[s:s+4096]) for s in range(0,len(idx),4096)])*sw[t]+mw[t]
        yt=Y[idx,t]; r2=1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
        print(f'{key:9s} k={k:2d} R2={r2:.4f}  result.json={ref[f"{key}|{k}"]:.4f}  N={len(idx)}')
        m=np.full(d[nm+'_U'].shape,np.nan); m[cj[idx],ci[idx]]=yp; out[f'{nm}_pred_{k}']=m
np.savez_compressed(os.path.join(CACHE, 'pred.npz'),**out)
