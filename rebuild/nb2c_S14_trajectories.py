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
def wrap(mod,orig):
    def f(hs,attention_mask=None,**kw): return orig(hs,attention_mask=attention_mask,past_key_values=cache,use_cache=True,**kw)
    return f
cache=Cache.from_legacy_cache(None)
for mod in mods: mod.forward=wrap(mod,GatedDeltaNet2.forward.__get__(mod))
tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
def erank(M):
    s=torch.linalg.svdvals(M.float().cpu()); s=s/(s.sum()+1e-12); return float(torch.exp(-(s*(s+1e-12).log()).sum()))
LAYERS=[4,8,12]; T=256; POS=[16,32,48,64,96,128,160,192,224,256]
def state_at(kw,t):
    sl=lambda x: x[:,:t] if torch.is_tensor(x) and x.dim()>=2 and x.shape[1]>=t else x
    o,st=orig_fr(q=sl(kw["q"]),k=sl(kw["k"]),v=sl(kw["v"]),g=sl(kw["g"]),b=sl(kw["b"]),w=sl(kw["w"]),
                 A_log=kw["A_log"],dt_bias=kw["dt_bias"],output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False)
    return st[0]
NAT=("Rivers shape the land over millions of years, carving valleys and depositing sediment far downstream. "
"The economy of a coastal town depends on fishing, tourism, and trade, each rising and falling with the seasons. "
"A good teacher notices when a student is confused before the student says a word. Winters in the north are long. "
"Music can carry a memory more vividly than a photograph, folding years into a single chord. History is a winding road. ")
MATH=("Let x be a positive integer. If 3x + 7 = 22 then x = 5. The sum 1 + 2 + ... + n equals n(n+1)/2. "
"The derivative of x^3 is 3x^2 and the integral of 2x dx is x^2 + C. A 3 4 5 triangle is right since 9 + 16 = 25. "
"Solve 2y - 4 = 10 to get y = 7. The probability of independent events multiplies. A prime has two divisors. ")
CODE=("def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n"
"class Stack:\n    def __init__(self): self.xs=[]\n    def push(self,x): self.xs.append(x)\n    def pop(self): return self.xs.pop()\n"
"for i in range(10):\n    if i%2==0: print(i)\n    else: continue\n")
KNOW=("The capital of France is Paris. Water boils at 100 degrees Celsius at sea level. Everest is the highest mountain. "
"The human heart has four chambers. The speed of light is about 299792 km per second. Photosynthesis converts sunlight. "
"DNA carries genetic information. The Great Wall of China spans thousands of kilometers. Gold is a dense metal. ")
DATA={"natural":NAT,"math":MATH,"code":CODE,"knowledge":KNOW}; NSEQ=4
res={}
for name,txt in DATA.items():
    big=tok(txt*12,return_tensors="pt").input_ids[0]
    seqs=[]
    for s in range(NSEQ):
        st=s*T
        if st+T<=big.shape[0]: seqs.append(big[st:st+T].unsqueeze(0).cuda())
    trajs=[]
    for ids in seqs:
        stash.clear(); cnt["i"]=0
        with torch.no_grad(): m(ids)
        tr=[]
        for t in POS:
            ers=[np.mean([erank(state_at(stash[li],t)[h]) for h in range(stash[li]["k"].shape[-2])]) for li in LAYERS]
            tr.append(np.mean(ers))
        trajs.append(tr)
    trajs=np.array(trajs); res[name]={"pos":POS,"mean":trajs.mean(0).tolist(),"std":trajs.std(0).tolist()}
    print("RESULT %s traj_mean_last=%.2f n_seq=%d"%(name,trajs.mean(0)[-1],len(seqs)),flush=True)
np.save("/root/nb2c_s14_data.npy",res,allow_pickle=True)
fig,ax=plt.subplots(figsize=(7,4.6)); cols={"natural":"#2471a3","math":"#c0392b","code":"#27ae60","knowledge":"#e67e22"}
for name in DATA:
    r=res[name]; mn=np.array(r["mean"]); sd=np.array(r["std"])
    ax.plot(POS,mn,"o-",color=cols[name],label=name); ax.fill_between(POS,mn-sd,mn+sd,color=cols[name],alpha=.15)
ax.set_xlabel("sequence position t"); ax.set_ylabel("state eRank(S_t)  (layers %s, head-mean)"%LAYERS)
ax.set_title("S14 (gdn2-1.3B): eRank trajectory by data type\n(mean ± std over %d sequences per type)"%NSEQ)
ax.legend(fontsize=9); plt.tight_layout(); plt.savefig("/root/nb2c_S14.png",dpi=150); print("PLOT_SAVED"); print("ALL_DONE")
