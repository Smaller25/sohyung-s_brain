import sys; sys.path.insert(0,"/root/dscpkg"); sys.path.insert(0,"/root/vfla")
import torch, warnings, numpy as np; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lit_gpt.gdn2 as G
from lit_gpt.config import Config; from lit_gpt.model import GPT; from lit_gpt.gdn2 import GatedDeltaNet2
from fla.models.utils import Cache; from transformers import AutoTokenizer
cfg=Config.from_name("gdn2_1.3B"); m=GPT(cfg).cuda().to(torch.bfloat16).eval()
ck=torch.load("/root/gdn2_1.3B_10B.pth",map_location="cpu",weights_only=False)
m.load_state_dict(ck["model"] if "model" in ck else ck,strict=False)
mods=[blk.attn for blk in m.transformer.h if getattr(blk,"use_gdn2",False)]
for i,mod in enumerate(mods): mod.layer_idx=i; mod.mode="fused_recurrent"
orig_fr=G.fused_recurrent_gdn2; stash={}; cnt={"i":0}
def cap(**kw):
    li=cnt["i"]%len(mods); cnt["i"]+=1
    stash[li]={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kw.items()}; return orig_fr(**kw)
G.fused_recurrent_gdn2=cap
cache=Cache.from_legacy_cache(None)
def wrap(mod,orig):
    def f(hs,attention_mask=None,**kw): return orig(hs,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
for mod in mods: mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod))
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
LAYERS=[6,10,14]
def erank(M):
    s=torch.linalg.svdvals(M.float().cpu()); s=s/(s.sum()+1e-12); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
def run_state(kw):
    o,st=orig_fr(q=kw["q"],k=kw["k"],v=kw["v"],g=kw["g"],b=kw["b"],w=kw["w"],A_log=kw["A_log"],dt_bias=kw["dt_bias"],
                 output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False); return st[0]
@torch.no_grad()
def fwd_capture(ids):
    stash.clear(); cnt["i"]=0; logits=m(ids); return logits
@torch.no_grad()
def S1_erank(ids):
    fwd_capture(ids)
    return float(np.mean([np.mean([erank(run_state(stash[li])[h]) for h in range(stash[li]["k"].shape[-2])]) for li in LAYERS]))
@torch.no_grad()
def density_bits(ids):
    lg=m(ids).float()[0]; p=torch.softmax(lg,-1); H=-(p*torch.log2(p+1e-12)).sum(-1); return float(H.mean())
def erank_curve(ids,start,stride=24):
    T=ids.shape[1]; ks=list(range(stride,T-start+1,stride)) or [T-start]
    vals=[S1_erank(ids[:,start:start+k]) for k in ks]; return np.array(ks),np.array(vals)
def segment_by_erank(ids,sat=0.9,stride=24,min_len=16,max_chunks=8):
    T=ids.shape[1]; start=0; bounds=[]
    while start<T-min_len and len(bounds)<max_chunks:
        ks,vals=erank_curve(ids,start,stride)
        if len(ks)==0: break
        reach=np.where(vals>=sat*float(vals.max()))[0]
        klen=max(int(ks[reach[0]]) if len(reach) else int(ks[-1]),min_len); start+=klen; bounds.append(start)
    return bounds
def chunk_lengths(bounds,T):
    b=[0]+bounds+([T] if (not bounds or bounds[-1]<T) else []); return [b[i+1]-b[i] for i in range(len(b)-1)]
# pool of real-ish prose (diverse)
POOL=("Rivers shape the land over millions of years carving valleys and depositing sediment far downstream. "
"The economy of a coastal town depends on fishing tourism and trade each rising and falling with the seasons. "
"A good teacher notices when a student is confused before the student says a word. In the north winters are long. "
"Music can carry a memory more vividly than a photograph folding years into a single chord. "
"The capital of France is Paris and water boils at one hundred degrees Celsius at sea level. "
"Photosynthesis converts sunlight into chemical energy stored in the bonds of sugar molecules. "
"A prime number has exactly two divisors and the sum of the first n integers is n times n plus one over two. "
"Trade routes once carried silk and spice across deserts and seas linking distant civilizations for centuries. ")*6
pool=tok(POOL,return_tensors="pt").input_ids[0].numpy()
L=256; DENS=[0.1,0.2,0.35,0.6,1.0]; NSEQ=4
def make_seq(uf,seed):
    rng=np.random.default_rng(seed); w=max(1,int(round(L*uf))); s=int(rng.integers(0,max(1,len(pool)-w)))
    win=pool[s:s+w]; seq=np.tile(win,int(np.ceil(L/w)))[:L]; return torch.tensor(seq,dtype=torch.long).unsqueeze(0).cuda()
import time; t0=time.time(); rows=[]
for uf in DENS:
    for s in range(NSEQ):
        ids=make_seq(uf,s); dens=density_bits(ids); bounds=segment_by_erank(ids); cl=chunk_lengths(bounds,L)
        rows.append({"uf":uf,"seed":s,"density":dens,"mean_chunk":float(np.mean(cl)),"n_chunks":len(cl)})
        print("RESULT uf=%.2f seed=%d density=%.2f mean_chunk=%.1f n=%d (%.0fs)"%(uf,s,dens,np.mean(cl),len(cl),time.time()-t0),flush=True)
np.save("/root/nb6_s21_data.npy",rows,allow_pickle=True)
d=np.array([r["density"] for r in rows]); mc=np.array([r["mean_chunk"] for r in rows])
from scipy.stats import spearmanr
try: rho,pv=spearmanr(d,mc)
except Exception: rho,pv=float("nan"),float("nan")
FIXED=float(np.mean(mc))  # density-blind fixed-split baseline (same avg budget)
fig,ax=plt.subplots(figsize=(6.6,4.4))
ax.scatter(d,mc,c="#c0392b",s=30,alpha=.7,label="eRank-saturation chunking (adaptive)")
# trend
o=np.argsort(d); z=np.polyfit(d,mc,1); ax.plot(np.sort(d),np.polyval(z,np.sort(d)),"--",color="#c0392b",lw=1)
ax.axhline(FIXED,color="#2471a3",lw=2,label="fixed-split baseline (same budget, density-blind)")
ax.set_xlabel("information density (bits/token)"); ax.set_ylabel("mean chunk length")
ax.set_title("S21: adaptive chunking tracks density; fixed split does not\n(eRank-saturation, gdn2-1.3B; Spearman rho=%.2f p=%.3f)"%(rho,pv))
ax.legend(fontsize=8.5); plt.tight_layout(); plt.savefig("/root/nb6_S21.png",dpi=150)
print("SPEARMAN rho=%.3f p=%.3f FIXED=%.1f"%(rho,pv,FIXED),flush=True); print("PLOT_SAVED"); print("ALL_DONE")
