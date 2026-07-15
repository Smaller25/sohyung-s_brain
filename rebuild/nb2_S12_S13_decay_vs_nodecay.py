import sys; sys.path.insert(0,"/root/vfla")
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fla.layers import DeltaNet, GatedDeltaNet2
from fla.models.utils import Cache
IGNORE=-100; T_FIX=384
def make_mqar(n,vocab,kv,seqlen,seed):
    rng=np.random.default_rng(seed); half=vocab//2
    kc=np.arange(1,half); vc=np.arange(half,vocab-1); ctx=2*kv; assert 4*kv<=seqlen
    inp=np.zeros((n,seqlen),dtype=np.int64); lab=np.full((n,seqlen+1),IGNORE,dtype=np.int64)
    for i in range(n):
        ks=rng.choice(kc,kv,replace=False); vs=rng.choice(vc,kv,replace=False)
        inp[i,0:ctx:2]=ks; inp[i,1:ctx:2]=vs
        gaps=np.sort(rng.choice((seqlen-ctx)//2,kv,replace=False))*2; qpos=ctx+gaps
        for j in range(kv): inp[i,qpos[j]]=ks[j]; lab[i,qpos[j]]=vs[j]
    return torch.tensor(inp),torch.tensor(lab[:,:seqlen])
def mk_mixer(kind,d,H,hd):
    if kind=="gdn2": return GatedDeltaNet2(hidden_size=d,head_dim=hd,num_heads=H,expand_v=1.0)
    return DeltaNet(hidden_size=d,num_heads=H,expand_k=1.0,expand_v=1.0,use_short_conv=True)  # no decay
class LM(nn.Module):
    def __init__(self,kind,vocab,d=128,L=2,hd=32,H=4):
        super().__init__(); self.embed=nn.Embedding(vocab,d)
        self.norms=nn.ModuleList([nn.RMSNorm(d) for _ in range(L)])
        self.mix=nn.ModuleList([mk_mixer(kind,d,H,hd) for _ in range(L)])
        for i,m in enumerate(self.mix): m.layer_idx=i
        self.nf=nn.RMSNorm(d); self.head=nn.Linear(d,vocab,False)
    def forward(self,ids):
        h=self.embed(ids)
        for n,m in zip(self.norms,self.mix): h=h+m(hidden_states=n(h))[0]
        return self.head(self.nf(h))
    @torch.no_grad()
    def erank_mean(self,ids):
        cache=Cache.from_legacy_cache(None); h=self.embed(ids)
        for i,(n,m) in enumerate(zip(self.norms,self.mix)):
            o,_,cache=m(hidden_states=n(h),past_key_values=cache,use_cache=True); h=h+o
        ers=[]
        for li in range(len(self.mix)):
            st=cache[li]["recurrent_state"]
            for b in range(min(st.shape[0],8)):
                for hh in range(st.shape[1]):
                    s=torch.linalg.svdvals(st[b,hh].float()); s=s/(s.sum()+1e-9); ers.append(float(torch.exp(-(s*(s+1e-12).log()).sum())))
        return float(np.mean(ers))
def run(kind,V=512,steps=5000):
    torch.manual_seed(0); dev="cuda"; m=LM(kind,V).to(dev).to(torch.bfloat16)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.1,betas=(0.9,0.95)); tk=[8,16,32]; t0=time.time()
    for step in range(1,steps+1):
        kv=tk[step%len(tk)]; ids,lab=make_mqar(32,V,kv,T_FIX,step); ids,lab=ids.to(dev),lab.to(dev)
        lg=m(ids).float(); loss=F.cross_entropy(lg.reshape(-1,V),lab.reshape(-1),ignore_index=IGNORE)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad()
        if step%2500==0: print("  [%s] step %d loss %.3f (%.0fs)"%(kind,step,loss.item(),time.time()-t0),flush=True)
    m.eval(); grid=[4,8,16,24,32,48,64,96]; rec=[]; er=[]
    with torch.no_grad():
        for kv in grid:
            ids,lab=make_mqar(96,V,kv,T_FIX,9000+kv); ids,lab=ids.to(dev),lab.to(dev); c=t=0
            for i in range(0,96,32):
                p=m(ids[i:i+32]).float().argmax(-1); msk=lab[i:i+32]!=IGNORE
                c+=(p[msk]==lab[i:i+32][msk]).sum().item(); t+=msk.sum().item()
            rec.append(c/max(t,1)); er.append(m.erank_mean(ids[:8]))
            print("RESULT %s kv=%d recall=%.3f erank=%.2f"%(kind,kv,rec[-1],er[-1]),flush=True)
    return grid,rec,er
res={}
for kind in ["deltanet","gdn2"]: g,rec,er=run(kind); res[kind]={"grid":g,"recall":rec,"erank":er}
np.save("/root/nb2_data.npy",res,allow_pickle=True)
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
cc={"gdn2":("#c0392b","GDN-2 (decay)"),"deltanet":("#2471a3","DeltaNet (no decay)")}
for k in ["deltanet","gdn2"]:
    c,lab=cc[k]; ax[0].plot(res[k]["grid"],res[k]["erank"],"o-",color=c,label=lab)
    ax[1].plot(res[k]["grid"],res[k]["recall"],"o-",color=c,label=lab)
ax[0].axhline(32,ls=":",color="gray",lw=.8); ax[0].text(96,32.5,"cap 32",ha="right",color="gray",fontsize=8)
ax[0].set_xlabel("MQAR load (# kv pairs), FIXED T=384"); ax[0].set_ylabel("state eRank"); ax[0].legend(fontsize=9)
ax[0].set_title("S12+S13: eRank rises with load; decay suppresses it\n(no-decay DeltaNet sits above decay GDN-2)")
ax[1].set_xlabel("MQAR load (# kv pairs), FIXED T=384"); ax[1].set_ylabel("recall"); ax[1].legend(fontsize=9); ax[1].set_ylim(-.03,1.05)
ax[1].set_title("recall vs load (same models)")
plt.tight_layout(); plt.savefig("/root/nb2_S12_S13.png",dpi=150); print("PLOT_SAVED"); print("ALL_DONE")
