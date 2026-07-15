import sys; sys.path.insert(0,"/root/vfla")
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, warnings, json
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fla.layers.gdn2 import GatedDeltaNet2
from fla.models.utils import Cache
IGNORE=-100
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
class GDN2LM(nn.Module):
    def __init__(self,vocab,d=128,L=2,hd=32,H=1):
        super().__init__(); self.embed=nn.Embedding(vocab,d)
        self.norms=nn.ModuleList([nn.RMSNorm(d) for _ in range(L)])
        self.mixers=nn.ModuleList([GatedDeltaNet2(hidden_size=d,head_dim=hd,num_heads=H,expand_v=1.0) for _ in range(L)])
        for i,m in enumerate(self.mixers): m.layer_idx=i
        self.normf=nn.RMSNorm(d); self.head=nn.Linear(d,vocab,False)
    def forward(self,ids):
        h=self.embed(ids)
        for norm,mix in zip(self.norms,self.mixers): h=h+mix(hidden_states=norm(h))[0]
        return self.head(self.normf(h))
    @torch.no_grad()
    def erank_mean(self,ids):
        cache=Cache.from_legacy_cache(None); h=self.embed(ids)
        for i,(norm,mix) in enumerate(zip(self.norms,self.mixers)):
            o,_,cache=mix(hidden_states=norm(h),past_key_values=cache,use_cache=True); h=h+o
        st=cache[len(self.mixers)-1]["recurrent_state"]  # [B,H,K,V]
        ers=[]
        for b in range(st.shape[0]):
            for hh in range(st.shape[1]):
                s=torch.linalg.svdvals(st[b,hh].float()); s=s/(s.sum()+1e-9); ers.append(float(torch.exp(-(s*(s+1e-12).log()).sum())))
        return float(np.mean(ers))
def main():
    torch.manual_seed(0); dev="cuda"; V=512
    m=GDN2LM(V,d=128,L=2,hd=32,H=1).to(dev).to(torch.bfloat16)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.1,betas=(0.9,0.95)); tk=[8,16,32,48]; t0=time.time()
    for step in range(1,6001):
        kv=tk[step%len(tk)]; ids,lab=make_mqar(32,V,kv,256,step); ids,lab=ids.to(dev),lab.to(dev)
        lg=m(ids).float(); loss=F.cross_entropy(lg.reshape(-1,V),lab.reshape(-1),ignore_index=IGNORE)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad()
        if step%1500==0: print("  train step %d loss %.3f (%.0fs)"%(step,loss.item(),time.time()-t0),flush=True)
    m.eval(); grid=[2,4,8,12,16,24,32,40,48,56,64,80,96]; rec=[]; er=[]
    with torch.no_grad():
        for kv in grid:
            sl=max(256, 4*kv); ids,lab=make_mqar(128,V,kv,sl,9000+kv); ids,lab=ids.to(dev),lab.to(dev)
            c=t=0
            for i in range(0,128,32):
                p=m(ids[i:i+32]).float().argmax(-1); msk=lab[i:i+32]!=IGNORE
                c+=(p[msk]==lab[i:i+32][msk]).sum().item(); t+=msk.sum().item()
            r=c/max(t,1); e=m.erank_mean(ids[:16])
            rec.append(r); er.append(e); print("RESULT kv=%d recall=%.3f erank=%.2f"%(kv,r,e),flush=True)
    np.save("/root/sweep_data.npy",{"grid":grid,"recall":rec,"erank":er,"cap":32},allow_pickle=True)
    # ---- plot: dual axis, gauge-not-knob ----
    fig,ax=plt.subplots(figsize=(8,5)); ax2=ax.twinx()
    l1=ax.plot(grid,rec,'o-',color="#1f77b4",lw=2,label="recall (accuracy)")
    l2=ax2.plot(grid,er,'s--',color="#d62728",lw=2,label="state eRank")
    ax2.axhline(32,color="#d62728",lw=.8,ls=":",alpha=.5); ax2.text(grid[-1],32.5,"d_state cap 32",color="#d62728",ha="right",fontsize=8)
    ax.set_xlabel("MQAR load (# key-value pairs)"); ax.set_ylabel("recall",color="#1f77b4"); ax2.set_ylabel("state eRank",color="#d62728")
    ax.set_ylim(-.03,1.05); ax2.set_ylim(0,34); ax.tick_params(axis='y',colors="#1f77b4"); ax2.tick_params(axis='y',colors="#d62728")
    # regime shading by recall
    rr=np.array(rec); g=np.array(grid)
    def band(mask,color,lab):
        xs=g[mask]; 
        if len(xs): ax.axvspan(xs.min(),xs.max(),color=color,alpha=.07)
    ax.axvspan(g[0],g[np.argmax(rr<0.9)] if (rr<0.9).any() else g[-1],color="green",alpha=.06)
    ax.set_title("eRank is a gauge, not a knob\n(GDN-2 hd32, MQAR load sweep — eRank climbs while recall collapses)",fontsize=11)
    ls=l1+l2; ax.legend(ls,[x.get_label() for x in ls],loc="center left")
    ax.annotate("recall cliff\n(capacity wall)",xy=(g[np.argmin(np.abs(rr-0.5))],0.5),xytext=(60,0.7),
                arrowprops=dict(arrowstyle="->",color="gray"),fontsize=9,color="gray")
    fig.tight_layout(); fig.savefig("/root/S18_erank_vs_recall.png",dpi=130)
    print("PLOT_SAVED",flush=True); print("ALL_DONE",flush=True)
main()
