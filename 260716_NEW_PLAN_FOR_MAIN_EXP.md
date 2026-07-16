# DynMC 1-Week Experiment Plan — Claude Code 실행 명세 (rev. 2)

> **목적**: Dynamic Memory Caching (DynMC) — GDN backbone 위에서 saturation/erasure 신호 기반 dynamic checkpoint placement — 의 main table 초안 + killer plot 2개 + trigger 신호 확정을 A100 4장, 7일 안에 산출.
> **rev. 2 변경**: SlimPajama-2k + long-anneal 2-stage → **ctx 16384 단일 stage (length-upsampled SlimPajama)**. Borrowed baseline row 전략 폐기 (Long-Data-Collections 공식 버전 takedown으로 DLA/LLA recipe 정확 재현 불가) → **자체 GDN anchor 학습 필수로 승격.**
> `[TO-FILL]` = 사용자 확인 필요 항목.

---

## 0. 프로젝트 컨텍스트

- **Method 한 줄**: 시퀀스를 independent-compressor segment로 나누고 (segment 경계에서 recurrent state reset), 각 segment의 최종 state를 cache에 보존, read는 online state + cached state들에 대한 output-space mixing. **Novelty = boundary를 고정 길이가 아니라 state-level 신호 (erasure/saturation) 로 동적 결정.**
- **핵심 대조**: MC-fixed (고정 길이) vs DynMC (dynamic trigger) — **같은 backbone weight에서 inference-time policy swap** (policy별 재학습 없음). No-cache GDN anchor는 동일 recipe 자체 학습.
- **Backbone**: Gated DeltaNet (GDN). Merge는 delta-rule state algebra (non-commutative transition Π(I − β_t k_t k_t^⊤)) 위반이므로 **checkpoint-only, no merge**가 방법론의 대수적 근거.
- **선행 대비**: MC (arXiv:2602.24281, fixed segmentation) / DLA (arXiv:2606.10650, drift 신호 + additive merge — commutative-gate 전용) / HOLA (arXiv:2607.02303, token-level KV cache). 신호 서사: flow (drift) / destruction (erasure) / capacity (saturation).
- **DLA 직접 비교는 차주**: 그들 코드를 본 mix 위에서 실행. 이번 주 main table은 self-contained controlled rows로 구성.

## 1. 환경 · Repo

- **Base**: `fla-org/flash-linear-attention` (GDN chunkwise kernel) + `fla-org/flame` (학습 파이프라인).
- **기존 코드**: `lmr/mosc/mosc_model.py` (segment cache, `_apply_cache_budget`), `utils.effective_rank`, `scripts/sh_axis2.sh` 참조. `[TO-FILL: 클러스터/VESSL 경로, 기존 GDN-2 d256/4L checkpoint 경로, wandb project]`
- **인프라 분담**: VESSL CPU run — 데이터 다운로드+tokenize → object storage (30GB binary). 로컬 — smoke test + G0 진단. VESSL A100 — Day 1 오전 G1부터.
- **Storage throughput 체크**: Day 1 G1에서 object-storage mount 직독 vs 인스턴스 로컬 NVMe 복사 후 학습의 step time 비교. 유의미한 차이 시 launch 스크립트에 "시작 시 binary를 NVMe로 복사" 추가.
- **실험 번호**: 0024부터.
- **커널 제약 (절대 준수)**: fla chunkwise kernel은 64-token chunk 경계마다 state를 materialize. **모든 boundary·신호 평가는 chunk 경계 (64의 배수) 에서만.** 커널 수정 금지 — checkpoint = materialized state 유지 wrapper.

## 2. Method 명세

### 2.1 Segment recurrence (independent-compressor 모드)

```
S_t = α_t S_{t−1} (I − β_t k_t k_t^⊤) + β_t v_t k_t^⊤ ,   S_{seg start} = 0
```

경계에서 state reset (chunk 경계 마스크). 완료된 segment의 최종 state S^(i)를 cache push.
**Packing 규칙**: 16k sequence에 문서 packing 시 **문서 경계 = 강제 segment 경계** (state reset). 문서 내부는 §2.3 random-length 분할.

### 2.2 Read (MC-GRM, output-space mixing)

```
y_t = γ_t^{(cur)} · S_t q_t + Σ_{i < cur} γ_t^{(i)} · S^{(i)} q_t
γ_t^{(i)} = softmax_i ⟨u_t, MeanPool(S^{(i)})⟩
```

- u_t: 학습되는 read projection. MeanPool 축·정규화는 mosc 기존 구현 컨벤션. `[TO-FILL #3]`
- Causality: 완료된 segment state만 attend. Gradient는 cached state 통과 (chunkwise autograd가 유지 — 메모리 프로파일만 확인).
- **Read 비용**: token당 ≈ L·H·K·(d_k·d_v) MAC. K=32에서 backbone 대비 ~20% overhead — 학습 시간 산정에 반영됨. K=64는 ~40%라 학습 중 금지.

### 2.3 학습 시 segment 길이: random-length

- log-uniform [64, 1024], 64의 배수 양자화. (문서 경계 reset과 중첩 적용)
- 근거: inference-time trigger의 길이 분포가 학습 support 안에 있어야 policy swap이 zero-shot 성립. Independent 모드에서 variable-length doc packing과 수학적 동일 → 학습 가능성 선례.
- **Fallback (G1 실패)**: curriculum — 처음 5B tokens fixed 256, 이후 random.

### 2.4 신호 정의 (chunk 경계 c; 학습 중 hook 로깅 + inference trigger)

Token t: α_t decay gate, β_t write gate, e_t = v_t − α_t k_t^⊤ S_{t−1}.

| 신호 | 정의 (chunk-누적) | 비고 |
|---|---|---|
| (a) drift | `‖S_c − S_{c−1}‖_F / (‖S_{c−1}‖_F + ε)` (보조: `Σ_t β_t‖e_t‖₂`도 로깅) | DLA 계열 baseline |
| (b) erasure | `E_c = Σ_t [ β_t‖α_t k_t^⊤ S_{t−1}‖₂ + (1−α_t)‖S_{t−1}‖_F ]` | overwrite항 + decay항. k^⊤S는 forward prediction 재활용 → 추가비용 ~0 |
| (c) saturation | `R_c = sr(S_c)/d_k`, `sr(S)=‖S‖_F²/‖S‖₂²` | ‖S‖₂: power iteration 5회, chunk 경계에서만 |
| (d) surprise (baseline) | `Σ_t ‖e_t‖₂/‖v_t‖₂` | HOLA Table 4에서 단독 실패 기지 — baseline 행 전용 |

Per-layer per-head 계산 후 head-mean. **Hook은 학습 시작 전부터 내장** (0-ii 재측정용).

### 2.5 Trigger policy (inference-time; matched density 필수)

- 규칙: 마지막 checkpoint 이후 누적 신호 > τ 시 checkpoint. Saturation은 레벨 초과 (`R_c ≥ τ_sat`).
- **Matched density**: 모든 policy 평균 밀도 1/N 동일 — held-out calibration set (val 10M tokens) 에서 τ quantile 캘리브레이션. **없으면 비교 무효.** 기본 N: 평균 segment 256 tokens.
- Policy: `mc-fixed`(256) / `drift` / `erasure` / `saturation` / `surprise`(baseline).

### 2.6 Budget 관리 (K 초과 시) — 0019 승격 ablation

- `drop-lowest`(default) / `fifo` / `merge-additive`(S_i ← S_i + S_{i+1}; 대수적 invalid 대조군). 예측: exact recall에서 drop ≥ fifo > merge.
- 학습 중 K=32 → 16k/256 = 64 segments라 **budget 정책이 학습 중 실제 발동** (의도된 동작 — 모델이 budget 하에서 read를 학습).

## 3. 데이터

### Main: length-upsampled SlimPajama (Fu et al., arXiv:2402.10171 방식)
- **SlimPajama 15.0B tokens, Mistral tokenizer (vocab 32000), ctx 16384.**
- **Upsampling**: source 비율 (CC/C4/GitHub/Books/arXiv/Wiki/SE) 은 원본 유지, **각 source 내부에서 긴 문서를 upsample** (Fu et al. per-source length-upsampling 그대로). Books ~4% + arXiv ~5%가 진짜 long-doc 신호원.
- 구현: tokenized binary에서 문서 길이 index 생성 → sampling weight 조정. 다운로드 추가 없음 (진행 중인 VESSL CPU run 산출물 사용).
- 짧은 문서는 16k로 packing, 문서 경계 = state reset (§2.1).
- **대안 (사용자 지시 시)**: FineWeb-Edu + PG19 + RedPajama-arXiv mix — 전부 공개 확인됨. 단 비율 자의성 + 추가 다운로드.
- 46M validation: FineWeb-Edu 0.5B (기존 계획 유지).

## 4. 학습 config

### 0024 — DynMC main (340M)
```
dmodel=1024, layers=24, heads=4, head_dim=256, expand_v=1, hidden_ratio=4,
conv=4, tied_embeddings=true, vocab=32000 (Mistral)
ctx=16384, data=upsampled SlimPajama 15B (§3)
+ DynMC: random-length independent segments, MC-GRM read, K=32, 신호 hook 내장
AdamW: peak_lr=4e-4, wd=0.01, cosine, warmup=1000, grad_clip=1.0
batch=0.5M tokens (= 32 seq × 16k), 1 epoch, bf16
checkpoint 매 1B tokens
```
- 예상: backbone 6ND ≈ 3.1e19 FLOPs + read ~20% → **~27–28h** (4×A100, MFU 30% 가정).
- **MFU 판정** (launch +2h): `tokens_per_sec × 86400 ≥ 15e9/1.4` (read overhead + 20% 마진). 미달 → R2.

### 0025 — GDN no-cache anchor (340M) **[필수]**
```
0024와 데이터/recipe/ctx 완전 동일, segment/cache/read 제거 (표준 GDN)
```
- ~23h. Day 2 저녁 launch (0024 완료 직후).

### 0024-pre — 46M validation
```
dmodel=512, layers=12, FineWeb-Edu 0.5B, ctx 4096
run A: random-length / run B: fixed-256
```

### 0026 — 170M scaling point (Day 5)
```
dmodel=1024, layers=12, 동일 데이터 6B tokens 슬라이스, ctx 16384
```
- ~5–6h (read overhead 포함).

## 5. Eval harness 명세

1. **LM**: Wikitext-103 test ppl, LAMBADA ppl/acc.
2. **NIAH position sweep** (killer plot 1 + mechanistic):
   - Context **16k** (+ 8k 보조), depth ∈ {0.05,...,0.95}, 위치당 ≥50 instance, teacher-forced accuracy.
   - 예측: mc-fixed sawtooth vs dynamic 평탄.
   - **Needle capture rate**: needle span [t_n, t_n+L]에 대해 boundary ∈ [t_n+L, t_n+L+128] 이고 해당 segment가 query 시점 cache 잔존한 비율. Policy별 capture↔recall instance 상관 보고.
3. **RULER**: S-NIAH-1/2/3, MK1, MV, MQ @ **2k/4k/8k/16k (in-length) + 32k/64k (extrapolation)** (Mistral tokenizer, limit 100). 64k는 buffer.
4. **Budget sweep**: K ∈ {8,16,32,64} × {drop, fifo, merge-additive}, NIAH 16k 기준.
5. **신호 진단 (실험 0)**: pairwise Pearson/Spearman + matched-density boundary Jaccard. 대상 (i) 기존 mosc checkpoint, (ii) 0024의 2–3B checkpoint.

## 6. Day-by-day

### Day 0 (오늘)
- [ ] VESSL CPU run: SlimPajama 다운로드 + Mistral tokenize → object storage (진행 중 확인). 완료 후 **문서 길이 index + upsampling weight 생성 잡** 이어 붙임.
- [ ] 로컬: 모델 코드 마감 (§2 전부) + 1B 서브셋 smoke test (tokenize→dataloader→46M 1 step).
- [ ] config 4종 작성 (0024/0025/0024-pre/0026).

### Day 1
- [ ] **오전 — G1**: 46M random vs fixed (VESSL A100 2장, 각 ~1h). **통과: random 최종 ppl 열화 ≤ 3%.** 실패 → curriculum fallback 후 재검증. 병행: storage throughput 체크 (§1).
- [ ] **오전 병렬 — G0 (0-i)**: 로컬에서 기존 checkpoint 신호 correlation + Jaccard. **판정: 전 쌍 Jaccard > 0.9 → STOP·사용자 보고 / 0.5–0.9 → 진행 + regime-conditional 분석 추가 / < 0.5 → 청신호.**
- [ ] **오후 — 0024 launch (4장, ~27h).** +2h MFU 판정.

### Day 2 (0024 학습 중)
- [ ] Eval harness §5 구현, 46M checkpoint로 검증.
- [ ] 0024 2–3B checkpoint로 0-ii 재측정.
- [ ] Trigger 캘리브레이션 (§2.5) 구현.
- [ ] **저녁 — 0024 완료 즉시 0025 (anchor) launch (~23h).**

### Day 3 (0025 학습 중)
- [ ] **Bake-off**: 5 policy × matched density, 0024 checkpoint, 전부 inference (0025 학습과 GPU 경합 시 anchor 3장 + bake-off 1장 split 허용, anchor ~30h로 연장 감수).
- [ ] Main table 초안 (anchor 행 제외 상태).

### Day 4
- [ ] 0025 완료 → anchor 행 추가, main table 완성.
- [ ] NIAH sawtooth plot + capture-rate figure.
- [ ] Budget sweep + drop/fifo/merge ablation.
- [ ] 승자 신호 잠정 확정 (NIAH 16k mean → RULER 8k–16k → ppl 순).

### Day 5
- [ ] 0026 (170M) launch (~6h) → scaling 행.
- [ ] 결과 정리, figure 확정.

### Day 6–7 (buffer)
- 재실행분 우선. 잉여 시: ① RULER 32k/64k extrapolation ② soft-gate 학습 버전 46M (H-Net ratio loss) ③ DLA 코드 확보·본 mix 실행 준비 (차주 본실행).

## 7. 리스크 · 사전 결정

| ID | 조건 | 대응 |
|---|---|---|
| R1 | G0 실패 (전 쌍 Jaccard > 0.9) | 모든 launch 중단, 사용자 보고 대기 |
| R2 | MFU 미달 (Day 1 +2h) | 0024 중단 → **170M/6B ctx 16k로 스케일 다운** (ctx·데이터 유지, 파라미터 축소). 340M 차주 큐잉 |
| R3 | G1 실패 | curriculum fallback → 46M 재검증 통과 시에만 0024 launch |
| R4 | loss spike/divergence | 직전 1B checkpoint에서 lr ×0.5 재개, 1회 한정. 재발 → 사용자 보고 |
| R5 | upsampling index 생성 지연 | Day 1 한정 uniform sampling으로 launch 불가 — **launch를 지연**하고 index 완성 우선 (데이터 분포가 실험의 전제이므로 타협 불가) |
| R6 | 0025 anchor가 Day 4까지 미완 | main table을 anchor 없이 제출용 초안으로 확정, anchor 행은 도착 즉시 삽입 (같은 recipe라 다른 행에 영향 없음) |

## 8. 산출물 체크리스트 (주말 기준)

- [ ] Main table: {GDN anchor(자체), mc-fixed, drift, erasure, saturation, surprise} × {Wiki, LMB, NIAH 16k, RULER 2k–16k}
- [ ] Killer plot 1: NIAH position sweep sawtooth (fixed vs dynamic)
- [ ] Killer plot 2: capture rate vs recall (mechanistic)
- [ ] 신호 진단 figure (correlation + Jaccard)
- [ ] Budget/정책 ablation (drop/fifo/merge × K)
- [ ] 170M scaling 행
- [ ] 승자 신호 메모 + 차주 계획 (DLA 본 mix 실행, RULER 64k, writing)

## 9. `[TO-FILL]`

1. ~~Stage 2 mix~~ → 해소 (upsampled SlimPajama 단일화). **대안 mix (FineWeb-Edu+PG19+arXiv) 로 갈지 여부만 확인** — default는 upsampled SlimPajama.
2. 클러스터/VESSL 경로, 기존 checkpoint 경로, wandb project명
3. MeanPool descriptor 축/정규화 (mosc 구현 확인)
4. 학습 K=32 승인 (read overhead ~20% 수반)
