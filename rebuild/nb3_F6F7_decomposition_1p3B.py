import sys; sys.path.insert(0,"/root/dscpkg"); sys.path.insert(0,"/root/vfla")
import torch, warnings, math; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lit_gpt.gdn2 as G
from lit_gpt.config import Config; from lit_gpt.model import GPT; from lit_gpt.gdn2 import GatedDeltaNet2
from fla.models.utils import Cache
from transformers import AutoTokenizer
cfg=Config.from_name("gdn2_1.3B"); m=GPT(cfg).cuda().to(torch.bfloat16).eval()
ck=torch.load("/root/gdn2_1.3B_10B.pth",map_location="cpu",weights_only=False)
m.load_state_dict(ck["model"] if "model" in ck else ck,strict=False)
mods=[blk.attn for blk in m.transformer.h if getattr(blk,"use_gdn2",False)]
for i,mod in enumerate(mods): mod.layer_idx=i; mod.mode="fused_recurrent"
print("n gdn2 layers",len(mods),flush=True)
orig_fr=G.fused_recurrent_gdn2; stash={}; cnt={"i":0}
def cap(**kw):
    li=cnt["i"]%len(mods); cnt["i"]+=1
    stash[li]={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kw.items()}; return orig_fr(**kw)
G.fused_recurrent_gdn2=cap
cache=Cache.from_legacy_cache(None)
def wrap(mod,orig,idx):
    def f(hs,attention_mask=None,**kw): return orig(hs,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
for i,mod in enumerate(mods): mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod),i)
NAT=("Rivers shape the land over millions of years, carving valleys and depositing sediment far downstream. "
"The economy of a coastal town often depends on fishing, tourism, and trade, each rising and falling with the seasons. "
"A good teacher notices when a student is confused before the student says a word. "
"In the north, winters are long and the light is thin; people learn patience from the weather. "
"Music can carry a memory more vividly than a photograph, folding years into a single chord. ")
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
ids=tok(NAT*4,return_tensors="pt").input_ids[:,:512].cuda()
with torch.no_grad(): m(ids)
G.fused_recurrent_gdn2=orig_fr
def erank(M):
    s=torch.linalg.svdvals(M.float().cpu()); s=s/(s.sum()+1e-12); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
def run_state(kw):
    o,st=orig_fr(q=kw["q"],k=kw["k"],v=kw["v"],g=kw["g"],b=kw["b"],w=kw["w"],A_log=kw["A_log"],dt_bias=kw["dt_bias"],
                 output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False); return st[0]
LAYERS=[int(round(x)) for x in np.linspace(2,len(mods)-2,6)]; LAYERS=sorted(set(LAYERS))
dk=stash[LAYERS[0]]["k"].shape[-1]; print("head_k_dim (cap)",dk,"layers",LAYERS,flush=True)
# ---------- F6: per-head r̄ vs erank ----------
rb=[]; er6=[]
for li in LAYERS:
    kw=stash[li]; g=kw["g"]; H=kw["k"].shape[-2]
    st=run_state(kw)
    for h in range(H):
        gh=g[...,h,:] if g.dim()>=3 else g
        rb.append(float(torch.exp(gh.float().mean()))); er6.append(erank(st[h]))
rb=np.array(rb); er6=np.array(er6)
fig,ax=plt.subplots(figsize=(6.2,4.6))
col=np.where(rb>=0.99,'#c0392b',np.where(rb>=0.9,'#e67e22','#2471a3'))
ax.scatter(rb,er6,c=col,s=22,alpha=.6)
xs=np.linspace(0.02,0.995,200); ax.plot(xs,np.minimum(dk,math.e/(1-xs)),'k--',lw=1,label='theory e/(1-r̄), cap %d'%dk)
ax.set_xlabel("r̄ = exp(E[log a_t])  (per head)"); ax.set_ylabel("effective rank of state")
ax.set_title("F6 (1.3B, natural text): decay vs erank per head"); ax.legend(fontsize=8); ax.set_ylim(0,min(dk,90))
plt.tight_layout(); plt.savefig("/root/F6_1p3B.png",dpi=170); np.save("/root/F6_1p3B_data.npy",{"rbar":rb,"erank":er6,"cap":dk})
print("F6 saved; erank range %.1f-%.1f cap %d"%(er6.min(),er6.max(),dk),flush=True)
# ---------- F7: 2x2 decomposition, NORM-MATCHED iso key ----------
torch.manual_seed(0)
conds={"real_g+real_k":(0,0),"g=1+real_k":(1,0),"real_g+iso_k":(0,1),"g=1+iso_k":(1,1)}
res={c:[] for c in conds}
for li in LAYERS:
    kw=stash[li]
    for c,(zg,rk) in conds.items():
        kw2=dict(kw)
        if zg: kw2["g"]=torch.zeros_like(kw["g"])
        if rk:
            r=torch.randn_like(kw["k"]); nr=r.norm(dim=-1,keepdim=True)+1e-8; nk=kw["k"].norm(dim=-1,keepdim=True)
            kw2["k"]=r*(nk/nr)                       # NORM-MATCHED iso key (direction random, norm preserved)
        st=run_state(kw2); res[c].append(np.mean([erank(st[h]) for h in range(st.shape[0])]))
mn={c:float(np.mean(v)) for c,v in res.items()}
dec=mn["g=1+real_k"]-mn["real_g+real_k"]; ani=mn["real_g+iso_k"]-mn["real_g+real_k"]
inter=mn["g=1+iso_k"]-mn["real_g+real_k"]-dec-ani
print("F7 real %.2f | +decay %.2f | +aniso %.2f | interaction %.2f | both-off %.2f"%(mn["real_g+real_k"],dec,ani,inter,mn["g=1+iso_k"]),flush=True)
fig,ax=plt.subplots(figsize=(5.4,3.9))
labels=list(conds); vals=[mn[c] for c in labels]
ax.bar(range(4),vals,color=["#2471a3","#e67e22","#c0392b","#7f8c8d"])
ax.set_xticks(range(4)); ax.set_xticklabels(["real g\nreal k","g=1\nreal k","real g\niso k\n(norm-matched)","g=1\niso k"],fontsize=8)
ax.set_ylabel("effective rank of state")
ax.set_title("F7 (1.3B, natural, norm-matched iso-k)\n+decay %.1f  +aniso %.1f  interaction %.1f"%(dec,ani,inter),fontsize=10)
for i,v in enumerate(vals): ax.text(i,v+0.15,"%.1f"%v,ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("/root/F7_1p3B.png",dpi=170); np.save("/root/F7_1p3B_data.npy",res)
print("F7 saved",flush=True); print("ALL_DONE",flush=True)
