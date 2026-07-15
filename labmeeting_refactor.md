# 랩미팅 발표 리팩터링 문서 — Dynamic Caching (2026-07-15)

- 대상 청중: AI 연구자이지만 아키텍처(SSM/linear attention) 비전공자.
- 원칙: (1) 사고 흐름 순서 그대로, (2) 모든 슬라이드에 [결과]/[진행 중]/[계획]/[아이디어] 딱지, (3) 수식보다 비유 먼저, 수식은 한 줄만.
- 이 문서의 용도: Claude Code에 넘겨서 (a) figure가 세팅 체크리스트를 만족하는지 검증, (b) 슬라이드 내용 채우기.

---

## 0. 발표의 한 문장과 세 기둥

**한 문장(첫 장 또는 끝 장에 박을 것):**
> Bounded-state 모델의 long-context 실패는 "용량이 작아서"가 아니다. (i) 용량을 늘리려는 multi-state 근사는 update rule의 대수를 위반하고(→ merging이 아니라 caching), (ii) state가 low-rank인 주원인은 decay이며(geometry는 부차적), (iii) 따라서 **eRank는 올리는 손잡이(knob)가 아니라 읽는 계기판(gauge)** — 계기판으로서의 용도(dynamic chunking 신호)가 살아남은 방향이다.

**세 기둥과 증거 슬라이드 매핑:**
| 기둥 | 주장 | 증거 |
|---|---|---|
| P1 대수 | additive multi-state는 delta 가족의 erasure를 위반 | S8 (F3 toy + DLA Table3), S9 (간접) |
| P2 진단 | low erank의 주범 = decay(+7.8), 부범 = key 뭉침(+1.5); eRank ≠ capacity | S15–S18 (F6–F9) |
| P3 개입 | erank를 올리는 개입(write-geometry)은 실전에서 실패 → gauge로 쓰는 chunking이 답 | S19 (0022/0023), S21–S22 |

**발표 전체의 punchline (S19에서 터뜨릴 것):** "eRank는 knob이 아니라 gauge다."
— 올리려고 하면(0023) 실패하고, 읽어서 경계 신호로 쓰면(RQ3) 유망하다. 이 한 문장이 RQ2→개입 실험→RQ3를 하나로 묶는다.

---

## 1. 청중 번역 장치 (비유 사전 — 슬라이드/대본에서 일관되게 사용)

| 개념 | 비유 | 사용 위치 |
|---|---|---|
| Recurrent state | 고정 크기 **화이트보드** (다 못 적으면 뭔가를 포기해야 함) | S3 |
| Linear attention (additive) | **포스트잇 쌓기** — 새 메모를 그냥 위에 얹음, 옛 메모 안 건드림 | S4 |
| Mamba-2 (scalar decay) | **바래는 잉크** — 전체가 매 스텝 같은 비율로 흐려짐 | S4 |
| Delta rule (GDN) | **지우고 다시 쓰는 화이트보드** — 같은 key 자리만 골라 지우고 새로 씀 | S4 |
| Multi-state caching | 화이트보드를 중간중간 **폴라로이드로 찍어** 보관 | S6 |
| 대수 위반 | **사진 속 글씨에는 지우개가 닿지 않는다** → 지운 값의 부활 | S8 |
| eRank | 창고 **선반 64칸 중 실제로 쓰는 칸 수** | S12 |
| decay에 의한 low rank | **시간이 지나면 짐이 저절로 사라지는 선반** — 오래된 칸이 비워짐 | S15–16 |
| key anisotropy | **짐을 같은 구석에만 쌓는 습관** — 칸이 남아도 못 씀 | S17 |
| knob vs gauge | 계기판(연료계)을 올린다고 연료가 늘지 않는다 | S19 |
| Hot state / cold KV | **해마(원본 에피소드) vs 신피질(통계 압축)** — CLS 이론, 본인 비유 유지 | S24 |

---

## 2. 구조 진단: 현재 24장 → 제안 구조 (가지치기)

### 유지하되 압축
- **S2–S6 Landscape 5장 → 3장.** 배경이 발표의 1/5를 먹으면 심장(RQ2, S19)이 밀린다.
  - S2 유지 (왜 아키텍처인가) — 단, hippocampus 직관은 여기서 빼고 S24로 이동 (중복 제거, 뇌 비유는 마지막에 한 번이 강함).
  - S3+S4 합침 → "recurrence = 압축 + update rule 3단 진화" 한 장 (포스트잇→바램→지우개).
  - S5 (hybrid) → **한 줄로 강등 또는 삭제.** hybrid는 이 발표의 본류가 아님. "hybrid가 뜨지만 왜 되는지 모른다 → 그래서 순수 recurrent의 원리부터"라는 한 문장이면 충분.
- **S6**: MoM은 한 줄만 ("write를 여러 방에 나누는 축 — 우리와 직교"). 지형은 2×2 사분면 하나로: boundary(고정/동적) × 용량 처리(파괴적 merge/비파괴 cache). 빈 칸 (동적, cache) = 본인 위치.

### 본문에서 빼고 백업으로
- **S9 (router 간접 증거)**: 본인 스스로 "번잡" 걱정 — 정확한 판단. 본문에서는 S8 끝에 한 문장("frozen backbone router에서도 single-key는 되고 multi-key는 무너지는 간접 증거가 있음, 자세한 건 백업")으로 처리하고 슬라이드는 백업으로 이동.
- **S11 testbed의 RetNet**: 실제 실험에 안 썼으면 목록에서 제거. 목록은 실제 사용한 것만 (Mamba-2, DeltaNet, GDN-1/2, vanilla LA).

### 중복/placeholder 정리
- **S20과 S23이 같은 placeholder 텍스트.** 역할 분리:
  - S20 = 경계 신호 후보 지형 (문헌: H-Net 코사인 라우팅 / DynLA drift / 본인: surprise, erank-plateau) — 표 하나.
  - S23 = RQ4 검증 **계획** (아래 §3-S23에 최소 실험 스펙 제공).
- **S19**: 현재 발표자 메모에 F9 경로가 잘못 복붙되어 있음. 여기가 0022/0023 자리 (§4).
- **S9의 발표자 메모도 S8 것 복붙 상태** (F3_delta_toy) — router figure로 교체 필요.

### 제목 제안 (선택)
현재: "Information Capacity-based expression for Linear Recurrence Models" — "expression"이 모호.
- 안 1: *State Capacity in Linear Recurrent Models: Diagnosis before Design*
- 안 2: *eRank is a Gauge, not a Knob: Capacity Diagnosis for Linear Recurrent Memory*
- 안 3 (한국어 부제 병기): *Linear recurrent 모델의 state 용량 — 진단, 개입, 그리고 dynamic caching*

---

## 3. 슬라이드별 정리

형식: **[딱지] 핵심 메시지** / 대본 / 슬라이드 내용 / Figure(경로·질문·세팅 체크리스트)

---

### S1. 제목 [—]
- **대본**: "오늘은 결과 발표라기보다, 한 학기 동안 linear recurrent 모델의 메모리를 파면서 제 사고가 어떻게 흘러왔는지, 어디서 틀렸고 어디로 피벗했는지를 정리해서 보여드리려 합니다. 관통 질문은 하나입니다 — 고정 크기 state는 왜, 어떻게 실패하는가."
- **내용**: 제목 + 한 문장 서사(§0) 미리보기(선택).

### S2. Landscape: 왜 아키텍처 수준인가 [동기]
- **메시지**: Agentic memory는 forward pass 밖의 해법 — "context 안에 이미 있는 정보를 state가 못 담는" 병목은 못 건드린다.
- **대본**: "요즘 memory 하면 다들 agentic memory — 외부에 텍스트를 저장하고 검색해서 다시 넣는 방식 — 를 떠올립니다. 그런데 검색해서 context에 다시 넣어도, 그 정보는 결국 **같은 고정 크기 state로 다시 압축**됩니다. 병목이 모델 안에 있으면 바깥 계층으로는 원리적으로 못 풉니다. 그래서 root cause, 즉 모델 설계부터 보기로 했습니다."
- **내용**: agentic(밖) vs architectural(안) 2단 다이어그램. 화살표: 검색→재주입→**재압축(같은 병목)**.
- **Figure**: 개념도 (데이터 불필요). 기존 F1 스펙.

### S3. Landscape: recurrence = 압축, update rule의 3단 진화 [배경] (구 S3+S4 합침)
- **메시지**: recurrent state는 "무엇을 남길지"를 매 토큰 결정하는 압축기이고, update rule의 역사는 더 나은 압축기를 향한 진화다.
- **대본**: "recurrent 모델은 context 전체를 고정 크기 화이트보드 하나에 요약합니다. 다 적을 수 없으니 매 토큰 '무엇을 포기할지' 결정해야 하죠. 그 결정 방식이 update rule이고, 세 세대로 진화했습니다. 1세대 linear attention은 **포스트잇** — 새 메모를 그냥 얹습니다. 옛 메모를 안 건드리니 겹치면 간섭이 생깁니다. 2세대 Mamba는 **바래는 잉크** — 전체가 조금씩 흐려져 최근 것이 선명합니다. 3세대 delta rule(GDN)은 **지우개 달린 화이트보드** — 같은 key 자리를 골라 지우고 새로 씁니다. 수정이 가능해진 거죠. 이 '지우개'가 오늘 이야기의 복선입니다."
- **내용**: 3단 비유 그림 + 수식 각 한 줄 (S←S+vkᵀ / S←aS+vkᵀ / S←S(I−βkkᵀ)+βvkᵀ). hybrid는 각주 한 줄.
- **Figure**: 3단 개념도 (새로 그리거나 손그림도 무방).

### S6(→새 S4). 문제 정의 + multi-state 지형 [배경→문제]
- **메시지**: long-context recall의 핵심 실패는 associative recall이고, 이를 풀려는 multi-state 계열은 2×2 지형에서 (동적, 비파괴 cache) 칸이 비어 있다.
- **대본**: "화이트보드가 하나면 용량이 부족하니, 최근 연구들은 보드를 여러 장 쓰기 시작했습니다. 쪼개는 **시점**이 고정이냐 내용 기반이냐, 그리고 오래된 보드를 **합쳐버리느냐(파괴) 사진으로 보관하느냐(비파괴)** — 이 두 축으로 정리하면: Log-linear은 (고정, 합침), DynLA는 (동적, 합침), Memory Caching은 (고정, 보관). **(동적, 보관) 칸이 비어 있고**, 그게 제 방향입니다. MoM은 '어느 보드에 쓸까'를 정하는 직교 축이라 오늘은 넘어갑니다."
- **내용**: 2×2 사분면 (F2). 세 논문 로고/이름 배치, 빈 칸에 별표.
- **Figure — F2 (새로 제작, 데이터 불필요)**:
  - 질문: "내 방향은 어디에 위치하는가?"
  - 세팅: 축 라벨 정확히 — x: boundary policy (fixed ↔ content-aware), y: old-state handling (destructive merge ↔ non-destructive cache).

### S7. 기존 방법의 한계 1: MC / Log-linear [문헌 분석·결과]
- **메시지**: 읽기 구조는 MC가 옳지만(비파괴 cache + gated read), 분할이 위치만 보고 내용을 안 본다.
- **대본**: "Memory Caching은 보관 방식은 옳습니다. 그런데 사진을 **256토큰마다 기계적으로** 찍습니다. 중요한 사건이 사진 중간에 걸리든 말든. 저자들도 결론에서 routing을 future work로 남겼습니다. Log-linear도 마찬가지로 위치 기반 고정 스케줄입니다. 즉, **언제 찍을지**가 열린 문제입니다."
- **내용**: MC 구조 한 컷 + "fixed segmentation (256) / routing = future work" 인용.

### S8. 기존 방법의 한계 2: DynLA의 대수 위반 [본인 분석·핵심]
- **메시지**: state를 쪼개 보관하는 순간 '나중 기록이 이전 기록을 고친다'는 delta의 기능이 스냅샷에 전달되지 않으며, 그 빠진 전달(행렬)은 스칼라 가중치로 원리적으로 복구 불가능하다.
- **대본**: "DynLA는 '언제'는 내용 기반으로 풀었는데, 보관을 **더하기로 합칩니다**. 여기서 문제가 생깁니다. 시나리오: 1구간에서 'CEO=A'를 기록하고 사진을 찍습니다. 5구간에서 delta rule이 CEO 자리를 지우고 B로 수정합니다. 그런데 **사진 속 A에는 지우개가 닿지 않습니다.** 읽을 때 사진과 현재 보드를 합치면 답이 A와 B의 혼합 — 지운 값이 부활합니다. '사진의 가중치를 낮추면 되지 않냐'고 할 수 있는데, 가중치는 사진 **전체**에 곱해지니 A를 죽이면 그 사진의 멀쩡한 기록도 같이 죽습니다. 스칼라로는 안 되는, 함수족 수준의 문제입니다. 이건 제 분석이고, 오른쪽 toy 실험으로 확인했습니다. 그리고 DynLA 자신의 수치에서도 흔적이 보입니다 — erase가 있는 GDN이 유일하게 multi-value에서만 Mamba-2에 지는데, 지운 값 보존이 '필요한' 태스크라서 그렇습니다."
- **내용**: (좌) CEO 시나리오 3컷 만화 or 다이어그램, (우) F3 bar + DLA Table3 재플롯.
- **Figure 1 — F3_delta_toy** (`report/labmeeting_figures/F3_delta_toy.png`):
  - 질문: additive multi-state read가 delta의 erasure를 실제로 무효화하는가? 연산자 합성 merge는 복구하는가?
  - 세팅 체크리스트:
    - [ ] delta rule, re-binding 쌍에 β=1 (완전 덮어쓰기), key L2-norm.
    - [ ] v_A ⊥ v_B (사영 분해가 유일하도록).
    - [ ] filler 쌍 포함 (스칼라 λ가 세그먼트1을 통째로 죽이는 자명해 방지).
    - [ ] 3조건: (a) 단일 state 끝까지, (b) 스냅샷+additive read — **λ는 최적 스칼라 허용** (근사에 최선 기회), (c) 연산자 합성 merge (S¹에 P₂=Π(I−βkkᵀ) 적용 후 합침).
    - [ ] y = 읽기 결과의 A/B 성분 스택 bar. 판독: (b)만 A 부활, (c)는 A≈0 복귀.
- **Figure 2 — DLA Table3 재플롯** (경로 미정, 새로 제작):
  - 질문: erase의 양날(다 이기는데 MV만 역전)이 공개 수치에 있는가?
  - 세팅: base Mamba-2 vs base GDN, S-NIAH-1/MK/MQ/MV grouped bar. **논문 수치 그대로** (MV: 19.4 vs 14.8), 출처 캡션 필수 (arXiv:2606.10650 Table 3).

### S9. → 백업 슬라이드로 이동 [부분 결과]
- 본문에서는 S8 대본 끝에 한 문장으로만. 백업 내용: frozen backbone + state-descriptor router, S-NIAH 32K 성공 / MK·MQ·MV 실패 figure.
- **Figure**: 발표자 메모가 현재 S8 것 복붙(F3_delta_toy) — **router 결과 figure로 교체 필요.** 세팅: 동일 backbone·동일 budget에서 task별 accuracy, "router가 무엇인지"는 캡션 한 줄로만.

### S10. Motivation: 압축 알고리즘의 문제인가, 용량의 문제인가 [전환]
- **메시지**: update rule(압축 알고리즘)을 더 고치는 경쟁 대신, 용량의 실체를 진단하고 → 유연하게 운용(dynamic caching)하는 쪽으로.
- **대본**: "여기까지가 문헌입니다. 그래서 저는 질문을 바꿨습니다. 다들 더 좋은 압축 알고리즘(update rule)을 만드는데 — 우리는 zip 알고리즘을 매번 새로 만들지 않죠. **압축률의 병목이 알고리즘인지, 아니면 용량 자체인지**부터 재야 하지 않나? 그래서 (1) state 용량이 실제로 얼마나 쓰이는지 진단하고, (2) 그 진단 신호로 '언제 사진을 찍을지'를 정하는 dynamic caching으로 가기로 했습니다. 남은 질문 세 개: 무엇이 좋은 신호인가(RQ1), 그 신호는 무엇을 재는가(RQ2), 신호로 경계를 정하면 작동하는가(RQ3-4)."
- **내용**: RQ 로드맵 한 장 (RQ1 신호 존재 → RQ2 신호의 의미 → [개입 실험] → RQ3 chunking → RQ4 검증).

### S11. Testbed [셋업]
- **메시지**: associative recall(키를 주면 값을 정확히 꺼내기)이 병목의 최소 재현이다.
- **대본**: "실험은 두 층입니다. 통제된 합성 태스크 MQAR — context에 key-value 쌍을 심고 나중에 key로 값을 묻는, 연상 기억의 최소 단위 — 와, 표준 벤치마크 RULER의 needle 계열. 모델은 update rule 축을 따라 vanilla linear attention / Mamba-2 / DeltaNet / GDN-1,2."
- **내용**: MQAR 예시 한 줄 (A→3, B→7, ..., "A?" → 3) + 모델 목록 (RetNet 제거).

### S12. RQ1: 저장량이 늘면 erank가 따라 오르는가 [결과]
- **메시지**: eRank(선반 사용 칸 수)는 외울 것이 늘수록 단조 증가 — load를 추적하는 신호일 후보.
- **대본**: "먼저 신호가 존재하는지. eRank는 '64칸 선반 중 실제로 쓰는 칸 수'입니다. MQAR에서 외워야 할 쌍 수만 늘리면 eRank가 따라 오릅니다. 즉 state가 얼마나 '차 있는지'를 밖에서 읽을 수 있다는 첫 신호입니다."
- **Figure — exp1.1** (`notebooks/capacity_results/worked_example_S1_D1_both.png`):
  - 질문: kv 쌍 수 ↑ → 최종 state eRank ↑ (단조)인가? plateau(포화점)가 있는가?
  - 세팅 체크리스트:
    - [ ] **시퀀스 길이 고정** (T가 변하면 erank가 T 때문에 오를 수 있음 — filler로 길이 맞춤).
    - [ ] 같은 모델·같은 체크포인트, kv 수만 조작.
    - [ ] eRank = entropy 정의 exp(H(σ/Σσ)), per-head 계산 후 평균, 최종 시점 state.
    - [ ] 축: x=kv 쌍 수, y=eRank(±head 표준편차).

### S13. RQ1: backbone에 따라 곡선이 달라지는가 [결과]
- **메시지**: decay가 있는 모델은 같은 load에서 erank가 눌린다 — update rule이 곡선을 결정.
- **대본**: "같은 실험을 update rule만 바꿔 반복하면 곡선이 갈립니다. decay가 있는 모델은 선반의 짐이 저절로 사라지니 같은 load에서 사용 칸이 적습니다. 이게 다음 섹션의 복선입니다 — low rank의 범인 후보 1번."
- **Figure — exp1.2** (`notebooks/decay_mqar_results/decay_mqar.png`):
  - 질문: decay 유/무 계열의 erank-load 곡선이 분리되는가?
  - 세팅 체크리스트:
    - [ ] 동일 task·d_state·학습 스텝·데이터, update rule만 교체.
    - [ ] decay 있는 계열(Mamba-2/GDN)과 없는 계열(vanilla LA/DeltaNet) 색 구분.
    - [ ] 같은 y축 스케일 (곡선 분리가 시각적으로 비교 가능).

### S14. RQ1: 실제 데이터에서도 신호가 변별력이 있는가 [결과]
- **메시지**: 합성 태스크 밖 — 데이터 종류(코드/자연어/구조화)에 따라 erank 궤적이 갈린다.
- **대본**: "합성만으론 부족하니 실제 텍스트에서 위치에 따른 erank 궤적을 봤습니다. 정보 밀도가 다른 데이터가 다른 궤적을 그립니다 — 신호가 내용을 구분한다는 뜻입니다."
- **Figure — exp1.3** (`notebooks/capacity_results/signal_trajectories_raw.png`):
  - 질문: erank(S_t) 궤적이 데이터 유형별로 분리되는가?
  - 세팅 체크리스트:
    - [ ] 같은 모델·토크나이저, 데이터 유형만 교체, 같은 T.
    - [ ] 유형별 여러 시퀀스 평균 ± 밴드 (한 시퀀스 일화 아님을 보장).
    - [ ] head-mean인지 특정 head인지 캡션 명시.

### S15. RQ2.1: 왜 이렇게 낮은가 — head별 분해 [결과]
- **메시지**: head마다 decay가 극단적으로 다르고(0.5↔0.99+), erank는 decay를 따라간다. 이론 곡선 min(d, e/(1−r̄))과 비교하면 decay가 설명하는 몫이 보인다.
- **대본**: "그런데 절대값이 이상합니다 — 최대 64칸인데 10~17칸. 왜? 먼저 head별로 쪼개 봤습니다. x축은 head의 평균 decay(기하평균), y축은 측정 erank. 회색 곡선은 'decay만 있고 key가 완벽히 퍼져 있다면'의 이론 상한 e/(1−r̄)입니다. 점들이 곡선을 따라가는 정도 = decay의 몫, 곡선 아래로 처지는 정도 = key 뭉침의 몫."
- **Figure — F6** (`report/labmeeting_figures/F6_erank_vs_decay.png`):
  - 질문: head별 erank가 decay-only 이론 곡선과 얼마나 정합/이탈하는가?
  - 세팅 체크리스트:
    - [ ] x = r̄ = exp(E[log aₜ]) — **산술평균 아님** (누적곱에 대응하려면 log-mean).
    - [ ] y = 측정 erank, **entropy 정의** (utils.effective_rank와 이론 상수 e/(1−r) 일치 확인; stable rank면 상수 ≈0.5/(1−r)로 곡선 교체).
    - [ ] 곡선 y = min(d_state, e/(1−x)) 오버레이, d_state 캡 표시.
    - [ ] 충분히 긴 고정 T (T ≫ 유효 horizon), 실제 데이터 forward.
    - [ ] head 유형(A/B/C) 색 구분(가능하면).

### S16. RQ2.1: 반사실 분해 — 범인 확정 [결과·중요]
- **메시지**: 학습 없이 forward 로그로 state를 반사실 재구성하면 두 요인이 가법 분해된다. **결과: decay가 지배(+7.8), key 뭉침은 부차(+1.5).** (실측 GDN2-370m 기준, 유휴 rank ~83%.)
- **대본**: "곡선 비교는 정황이니 인과로 확정합니다. forward에서 (decay, key, value)를 전부 로그해두고 state를 네 가지로 다시 조립합니다 — decay를 켜고/끄고 × key를 실제/이상적으로 퍼진 랜덤으로. 그러면 각 요인이 erank를 얼마나 깎는지 **가법적으로** 분리됩니다. 결과: **decay가 주범입니다** (+7.8), key 뭉침은 부차적입니다 (+1.5). 솔직히 저는 key 뭉침이 주범일 거라 예상했는데, 데이터가 아니라고 했습니다. 이 결과가 뒤의 개입 실험 해석을 전부 결정합니다."
- **Figure — F7** (`report/labmeeting_figures/F7_decomposition.png`):
  - 질문: low erank에서 decay 몫과 anisotropy 몫은 각각 얼마인가?
  - 세팅 체크리스트:
    - [ ] forward에서 (aₜ, kₜ, vₜ) 로그 → 2×2 재구성: {실제 decay, aₜ≡1} × {실제 key, 등방 랜덤 key}.
    - [ ] **등방 key는 실제 key와 norm 분포 매칭** (방향만 랜덤화; norm까지 바꾸면 교란변수).
    - [ ] value stream 동일 유지.
    - [ ] 네 조건 erank + 가법 분해값(+7.8 / +1.5) 표기, **상호작용항** 크기 확인·보고.
    - [ ] 실측 모델·데이터 명시 (GDN2-370m, 어떤 코퍼스, T).

### S17. RQ2.1: 뭉침의 직접 증거 [결과·보조]
- **메시지**: 부차적이지만 실재한다 — key들이 등방 대비 유의하게 뭉쳐 있다.
- **대본**: "부범도 실재합니다. key 쌍의 코사인 유사도 분포가 등방 기준(회색)보다 오른쪽으로 치우쳐 있습니다 — 짐을 같은 구석에 쌓는 습관. 다만 F7이 보여주듯 이건 두 번째 요인입니다."
- **Figure — F8** (`report/labmeeting_figures/F8_concentration.png`):
  - 질문: key 방향이 등방 null 대비 뭉쳐 있는가?
  - 세팅 체크리스트:
    - [ ] head별 key 쌍별 코사인 히스토그램 + **같은 차원 등방 랜덤 null 분포 오버레이**.
    - [ ] 충분한 표본 수(쌍 수) 캡션 명시.

### S18. RQ2.2: erank는 '용량 계기판'으로 유효한가 [결과·전환점]
- **메시지**: 조건부로만. erank와 recall은 **state가 실제로 과적재일 때만** 함께 움직인다 → **eRank ≠ capacity, 계기판이지 용량 자체가 아님.**
- **대본**: "그럼 erank가 높으면 성능이 좋은가? 데이터 유형·적재 상태별로 보면 **아닙니다.** 여유 구간에선 erank를 올려도 recall이 안 움직이고, 과적재 구간에서만 둘이 같이 움직입니다. 결론: eRank는 용량 그 자체가 아니라, **적재 상태에서만 의미 있는 계기판**입니다. 그럼 자연스러운 다음 질문 — 계기판이 아니라 진짜 용량을 늘릴 수는 없나? 그래서 개입 실험을 했습니다."
- **Figure — F9** (`report/labmeeting_figures/F9_datatypes.png`):
  - 질문: erank–recall 관계가 regime(여유/포화/과적재)에 따라 달라지는가?
  - 세팅 체크리스트:
    - [ ] regime 정의를 축이나 패널로 명시 (load 또는 state 크기로 조작).
    - [ ] erank와 recall을 같은 조건에서 짝지어 측정 (교차 조건 비교 금지).

### S19. 개입 실험: erank를 올리면 성능이 오르는가 [결과·negative·punchline]
→ §4에 전체 스펙. 여기가 0022/0023 자리. **현재 발표자 메모의 F9 경로는 잘못 복붙 — 교체.**

### S20. RQ3: 경계 신호 후보 지형 [문헌+아이디어]
- **메시지**: '언제 사진을 찍을까'의 후보 — 문헌(H-Net 코사인, DynLA drift)과 본인 후보(surprise, **erank-plateau**).
- **대본**: "개입 실험이 알려준 대로, erank는 올리는 게 아니라 **읽는** 겁니다. 읽어서 뭘 하나 — '언제 사진을 찍을지'를 정합니다. 후보 신호는 넷: 문헌에서 H-Net의 인접 표현 코사인, DynLA의 state drift. 제 후보는 delta rule이 공짜로 주는 prediction error(surprise), 그리고 **erank가 plateau에 닿는 순간**(선반이 다 찼다는 계기판 신호)입니다."
- **내용**: 4후보 비교 표 (신호원: 입력측/상호작용/상태측, 비용, 실패 메커니즘과의 연결).

### S21. RQ3: chunking 결과 1 [진행 중]
- **Figure — exp3.1a** (`notebooks/chunking_results/chunk_by_density.png`):
  - 질문: 정보 밀도/신호 기반 경계가 고정 분할과 다르게, 의미 있게 찍히는가?
  - 세팅 체크리스트:
    - [ ] 비교 조건에 **고정 분할 baseline** 포함, **동일 chunk 수(budget)** 통제.
    - [ ] 신호 종류(밀도? erank? drift?) 캡션 명시.

### S22. RQ3: chunking 결과 2 — worked example [진행 중]
- **Figure — exp3.1b** (`notebooks/chunking_results/worked_example_boundaries.png`):
  - 질문: 경계가 실제 시퀀스의 의미 전환/needle 위치와 정렬되는가?
  - 세팅 체크리스트:
    - [ ] 실제 시퀀스 텍스트 위 경계 마커 + ground-truth 전환점(topic 경계/needle 위치) 표시.
    - [ ] 체리피킹 방지: 대표 예시 + "무작위 추출 N개 중 정렬률" 수치 한 줄.

### S23. RQ4: 작동하는가 — 검증 계획 [계획]
- **메시지**: 최소 검증 스펙이 준비되어 있다 (아직 결과 없음 — 솔직하게).
- **내용 (채울 것)**:
  - Task: segment 구조 MQAR (topic block + 경계에서 key 분포 점프; re-binding 변형 포함 — S8의 대수 예측을 직접 시험).
  - 조건: frozen backbone + MC-GRM read, 경계만 {erank-plateau / drift / 고정 / 랜덤} 교체, **동일 checkpoint budget**.
  - 성공 기준: 동일 budget에서 needle-위치별 accuracy 곡선이 고정 분할의 톱니(sawtooth)를 평탄화하면 성공.
  - 이후: RULER 4K→16K 확장, from-scratch joint 학습(자원 확보 시).
- **대본**: "여긴 아직 빈 칸입니다. 다만 무엇을 하면 판가름 나는지는 정해져 있습니다 — 같은 사진 예산에서 경계 규칙만 바꿔서, needle 위치별 정확도의 톱니가 펴지는지 봅니다."

### S24. Further: Hot state / Cold KV + CLS [아이디어]
- **메시지**: 압축이 안 되는 '구체적' 정보는 state에 흔적만 남기고 원본은 dictionary로 — 해마(에피소드 원본)/신피질(통계 압축)의 CLS 구조와 동형.
- **대본**: "마지막 아이디어입니다. 한 달 지나면 '그 책에서 그 종을 봤다'는 **흔적**은 남는데 정확한 내용은 안 나죠. 그 정확한 내용이 필요한 태스크가 needle 계열입니다. 그래서: 압축이 잘 되는 구조적 정보는 state가 담고, 압축이 안 되는 구체적 항목(UUID, 숫자)은 **써지는 길목에서 원본을 dictionary로 복사**해두는 겁니다 — state를 거꾸로 복호하는 게 아니라요. 이건 해마/신피질의 상보 학습(CLS) 구조와 정확히 같은 분업이고, Transformer 쪽 KV-cache 계층화(hot GPU/cold host)에선 이미 검증된 패턴인데 bounded-state 쪽엔 아직 없습니다."
- **내용**: hot(GPU state, O(K)) / cold(host dictionary) 계층 개념도 + "spill = write 길목 복사(복호 아님)" 명시. 정보 기준 각주: 후보 선정 신호는 '압축 불가 잔여'(time-bounded entropy) 쪽 — epiplexity(구조적 정보)의 보완 개념 (arXiv:2601.03220 참고).
- **(선택) 싼 첫 실험**: MK/MV NIAH에서 state만 vs state+oracle dictionary — headroom 상한 측정.

---

## 4. 슬라이드 19 채우기: 0022/0023 (개입 실험, negative result)

### 스토리 (대본)
"RQ2가 '뭉침이 부차적'이라 했지만, 그래도 확인해야 했습니다 — **놀고 있는 83%의 칸에 강제로 나눠 쓰게 하면(write를 펼치면) recall이 오르나?** 두 단계로 시험했습니다.

1단계, decay를 **끈** 장난감 delta-net에서 (뭉침 요인만 고립): 여유 구간에선 효과 없음, 포화 직전엔 오히려 해로움, **과적재에서만** 효과 — kv64에서 recall 0.08→0.18, 2.2배. 조건부로는 됩니다.

2단계, **진짜 GDN-2** (decay·erase 게이트 있는 실물)에 이식: **전 조건에서 실패**했습니다. recall이 모든 설정에서 떨어지고, 규제를 약하게 해도(λ=0.1) 방향 자체가 해로우며, erank는 오르기는커녕 16→3으로 **붕괴**합니다. 장난감에서 유일한 rank 제한자였던 뭉침이, 실물에선 decay에 종속된 2차 요인이라서입니다 — key를 억지로 펼치면 모델이 학습해둔 key↔gate 결합과 싸우게 됩니다.

결론이자 오늘의 punchline: **eRank는 knob이 아니라 gauge입니다.** 계기판을 손으로 밀어올린다고 연료가 늘지 않습니다. 설계 레버는 write geometry가 아니라 **decay/retention**이고, gauge로서의 erank는 다음 섹션(chunking 신호)에서 삽니다. 이 negative가 제 가지치기입니다 — idea A는 접고, retention과 idea C로 갑니다."

### 슬라이드 내용
- 좌 패널: **0022 (decay-free toy)** — regime R1/R2/R3 × {baseline, C4} recall bar. R3(kv64)에 "2.2×" 강조.
- 우 패널: **0023 (real GDN-2)** — 3 config × {baseline, C4} recall bar + erank 수치 (16.2→6.2/2.8) 작게 병기. 전부 하락 → 붉은 색.
- 하단 한 줄: "eRank is a gauge, not a knob → design lever = retention; gauge 용도 = chunking 신호 (다음 장)".
- 딱지: [결과 — negative, toy scale·single seed 캐비앗 명시]

### Figure 세팅 체크리스트 (새로 제작; 데이터는 0022/0023 RESULT 라인)
- [ ] 0022 패널: R1(H2×64, kv 8/16/32) / R2(H1×32, kv 32/48/64) / R3(H1×16, kv 32/48/64) 구분 명확히; baseline vs C4만 (C1/C2는 백업).
- [ ] 0023 패널: hd32·λ0.5 / hd32·λ0.1 / hd16·λ0.5 세 config; recall kv32/48/64.
- [ ] 두 패널 y축 스케일 통일 여부 결정 (권장: 패널별 자유 + 수치 라벨).
- [ ] 캡션에 세팅 요점: 0022는 aₜ≡1 (뭉침 요인 고립·직접 구현 delta-net 2층·MQAR·2000스텝), 0023은 FLA GatedDeltaNet2 (실물 decay/erase·hook만으로 C4 이식·5000스텝·phase transition 고려), 둘 다 single A100·single seed.
- [ ] 정직 캐비앗 각주: toy scale, C4만 실전 이식(C1/C2는 kernel 수정 필요), idea A 계열 전체의 완전 배제는 아님.

### 예상 질문 방어 (발표자 메모용)
- "λ 튜닝 부족 아닌가?" → λ=0.1에서도 recall 하락 + erank가 **더** 붕괴(2.81) — 과규제가 아니라 방향의 문제.
- "왜 C4만 실물 테스트?" → 0022 승자이자 loss-only라 kernel 무수정 이식 가능한 가장 깨끗한 시험; C1/C2는 같은 학습된 기하와 싸우므로 더 나쁠 개연성.
- "그럼 retention 개입은 뭘 할 건가?" → 중요한 방향/토큰의 aₜ→1 유지(selective retention), GDN-2의 게이팅과 **함께** 가는 설계 — 다음 학기 1순위.

---

## 5. Figure–경로–세팅 총괄표 (Claude Code 검증용 요약)

| 슬라이드 | 경로 (발표자 메모 추출) | 답하는 질문 | 핵심 검증 포인트 |
|---|---|---|---|
| S8 | `report/labmeeting_figures/F3_delta_toy.png` | additive read가 erasure를 무효화하나 | β=1, v_A⊥v_B, λ 최적 허용, (c) 연산자 합성 조건 포함 |
| S8 | (신규) DLA Table3 재플롯 | MV 역전의 실재 | 논문 수치 그대로 + 출처 |
| S9(백업) | (교체 필요 — 현재 F3 복붙) | router의 single vs multi-key 격차 | 동일 budget, task별 accuracy |
| S12 | `notebooks/capacity_results/worked_example_S1_D1_both.png` | load↑→erank↑? | **시퀀스 길이 고정**, entropy 정의, head 평균 |
| S13 | `notebooks/decay_mqar_results/decay_mqar.png` | decay 유무로 곡선 분리? | rule만 교체, 동일 d_state·스텝 |
| S14 | `notebooks/capacity_results/signal_trajectories_raw.png` | 데이터 유형별 궤적 분리? | 유형별 다시퀀스 평균±밴드 |
| S15 | `report/labmeeting_figures/F6_erank_vs_decay.png` | decay-only 곡선과의 정합/이탈 | r̄=exp(E[log a]), erank 정의↔상수 일치, min(d,·) 캡 |
| S16 | `report/labmeeting_figures/F7_decomposition.png` | decay 몫 vs 뭉침 몫 | 등방 key norm 매칭, value 고정, 상호작용항 보고 |
| S17 | `report/labmeeting_figures/F8_concentration.png` | 뭉침의 직접 증거 | 등방 null 오버레이, 표본 수 |
| S18 | `report/labmeeting_figures/F9_datatypes.png` | erank–recall의 regime 의존성 | regime 명시, 동일 조건 짝짓기 |
| S19 | (신규 — 메모의 F9 경로는 오기) | erank를 올리면 성능↑? (아니오) | §4 체크리스트 |
| S21 | `notebooks/chunking_results/chunk_by_density.png` | 신호 기반 경계 vs 고정 | 고정 baseline + 동일 chunk 수 |
| S22 | `notebooks/chunking_results/worked_example_boundaries.png` | 경계–의미 전환 정렬 | ground-truth 전환점 표시, 정렬률 수치 |

---

## 6. 가지치기 최종 목록 (한눈에)
1. Landscape 5장 → 3장 (S3+S4 합침, S5 hybrid 한 줄 강등).
2. Hippocampus 직관: S2에서 제거, S24에서만.
3. S9 router: 본문 한 문장 + 백업 이동. 발표자 메모 figure 경로 교체.
4. S11: RetNet 제거 (미사용 시).
5. S20/S23 placeholder 중복 해소: S20=신호 지형, S23=검증 계획.
6. S19: F9 오기 경로 삭제, 0022/0023로 채움 (§4).
7. MoM: S6에서 한 줄 처리.
8. idea A(write-geometry)는 발표에서 "접은 가지"로 명시 — 가지치기 자체를 서사로.
