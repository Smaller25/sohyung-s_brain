"""nb1 / S8-F3 — additive merge violates the delta algebra (fixed version).
Adds vs original: (i) FILLER key-value pairs, (ii) additive read with the OPTIMAL scalar
weight (best chance), (iii) condition (c) operator-composition merge P=(I-k kᵀ).
Point: a SCALAR-weighted additive read faces a dilemma — killing the overwritten value A
also kills segment-1 filler; only the MATRIX operator (c) removes A while keeping filler.
Our analysis, no model, CPU/numpy."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(0); d=32; m=4                      # m filler pairs (in segment 1)
# orthonormal keys: k (re-bound) + f_1..f_m (filler), via QR
K=rng.standard_normal((d,m+1)); Q,_=np.linalg.qr(K); keys=Q[:,:m+1]
k=keys[:,0]; F=keys[:,1:]                                    # k=shared key, F=filler keys
# orthonormal-ish values for clean readout: A,B (bound to k across segments) + filler values g_j
V=rng.standard_normal((d,m+2)); Qv,_=np.linalg.qr(V); A=Qv[:,0]; B=Qv[:,1]; G=Qv[:,2:2+m]
def write(S,kk,vv,beta=1.0): return S-beta*np.outer(S@kk,kk)+beta*np.outer(vv,kk)  # delta (unit key)
# --- segment states ---
# seg1: bind k->A AND filler f_j->g_j ;  seg2: RE-BIND k->B (the overwrite)
S1=np.zeros((d,d)); S1=write(S1,k,A)
for j in range(m): S1=write(S1,F[:,j],G[:,j])
S2=write(np.zeros((d,d)),k,B)
# (a) single continuous state over both segments (seg1 writes, then seg2 overwrite)
Sc=S1.copy(); Sc=write(Sc,k,B); read_a=Sc
# (b) additive read w1*S1+w2*S2 ; (c) operator merge S2 + S1@P, P=(I-k kᵀ)
P=np.eye(d)-np.outer(k,k)
def rd(S,x): return S@x
def acomp(S): return abs(rd(S,k)@A)
def bcomp(S): return abs(rd(S,k)@B)
def filler(S): return float(np.mean([rd(S,F[:,j])@G[:,j] for j in range(m)]))  # want ~1
# (b) additive with EQUAL weights and with the OPTIMAL scalar w1 that minimizes |A| (=0)
def add(w1,w2): return w1*S1+w2*S2
b_eq=add(1,1)                                # equal weights
b_killA=add(0,1)                             # only scalar that removes A  -> destroys filler
merge_c=S2+S1@P                              # (c) operator-composition merge
conds=[("single\n(continuous)",read_a),("additive\nscalar w=1",b_eq),
       ("additive\nw1→0 (kill A)",b_killA),("operator merge\nS2 + S1·(I-kkᵀ)",merge_c)]
Acs=[acomp(S) for _,S in conds]; Bcs=[bcomp(S) for _,S in conds]; Fcs=[filler(S) for _,S in conds]
# --- plot ---
fig,ax=plt.subplots(figsize=(7.4,4.0)); x=np.arange(len(conds)); w=0.26
ax.bar(x-w,Acs,w,label="A  (overwritten — want 0)",color="#c0392b")
ax.bar(x  ,Bcs,w,label="B  (current — want 1)",color="#2471a3")
ax.bar(x+w,Fcs,w,label="filler recall (want 1)",color="#27ae60")
ax.set_xticks(x); ax.set_xticklabels([c for c,_ in conds],fontsize=8.5)
ax.set_ylabel("|read · value|"); ax.set_ylim(0,1.2)
ax.set_title("Only operator-composition merge removes the deleted value A\n"
             "(scalar-additive can't: killing A also kills segment-1 filler)",fontsize=10.5)
ax.legend(fontsize=8,frameon=False,ncol=1,loc="upper right")
for xi,(a_,b_,f_) in enumerate(zip(Acs,Bcs,Fcs)):
    for off,val in [(-w,a_),(0,b_),(w,f_)]: ax.text(xi+off,val+0.02,f"{val:.2f}",ha="center",fontsize=7)
plt.tight_layout(); out="/home/sohyung/sohyung's_brain/rebuild/F3_delta_toy_fixed.png"
plt.savefig(out,dpi=200)
print("A:",[f"{v:.2f}" for v in Acs]); print("B:",[f"{v:.2f}" for v in Bcs]); print("filler:",[f"{v:.2f}" for v in Fcs])
print("saved",out)
