# FINAL REPORT — Linear-recurrent 모델의 state 용량: 진단 · 개입 · 신호 (2026-07-24)

이 문서는 지금까지의 동기, 각 실험의 목적·세팅·**보수적 결과**(부풀리지 않음), 그리고 그로부터 방어
가능한 **객관적 사실**만 정리한다. 수치는 실측 그대로이고, 각 실험마다 한계(캐비앗)를 함께 적는다.

---

## 1. 배경과 동기

- Mamba-2, Gated DeltaNet(GDN) 같은 linear-recurrent 모델은 context 전체를 **고정 크기 recurrent
  state** 하나에 압축한다. 이게 attention 대비 효율(O(1) state)의 원천이다.
- 고정 크기라 저장할 수 있는 연상(key→value)의 수에 상한이 있다. context가 길어져 key가 많아지면
  간섭이 생겨 long-context recall이 무너진다.
- **질문**: 효율(≈O(1))을 포기하지 않고 이 모델의 실효 recall을 늘릴 수 있는가?
- 방향 전환: 더 나은 update rule(압축 알고리즘)을 새로 만드는 경쟁 대신, **state 용량의 실체를
  먼저 진단**하고 → 그 진단 신호를 설계에 쓰자. 진단의 중심 지표 후보가 **effective rank(eRank)**
  이다. eRank = state 행렬의 특이값 분포 엔트로피의 exp = "state가 실제로 쓰는 차원 수".

핵심적으로 답해야 할 것 세 가지:
1. eRank가 부하(외울 양)를 따라가는 신호인가? (RQ1)
2. eRank가 낮게 유지되는 **원인**은 무엇인가? (RQ2 — decay vs key 뭉침)
3. eRank는 **용량 그 자체**인가, 아니면 **읽기만 하는 계기판**인가? 올리는 개입이 성능을 올리나? (RQ3)

---

## 2. 실험 총괄표

| # | 파일 | 목적 | 모델 | 핵심 결과(보수적) |
|---|---|---|---|---|
| E0 | `rebuild/nb1_F3_algebra.py` | multi-state를 더하기로 합치면 delta의 erasure가 깨지는가 | 수식 toy(numpy) | additive는 지운 값 부활, 연산자-합성 merge만 정상 |
| E1 | `rebuild/nb2_S12_S13_decay_vs_nodecay.py` | eRank가 부하 따라 오르나 / decay가 eRank를 누르나 | from-scratch 2-layer (GDN-2, DeltaNet) | 둘 다 부하↑→eRank↑; no-decay가 decay보다 eRank 약 2배 높음 |
| E2 | `rebuild/nb2c_S14_trajectories.py` | 데이터 유형별 eRank 궤적이 갈리나 | gdn2-1.3B | 유형 간 궤적 **크게 겹침**(분리 약함) |
| E3 | `rebuild/nb3_F6F7_decomposition_1p3B.py` | 낮은 eRank의 원인 분해(decay vs 뭉침) | gdn2-1.3B | decay 몫(+43.7)이 뭉침 몫(+9.9)보다 지배적 |
| E4 | `rebuild/nb4_S18_erank_gauge.py` | eRank와 recall이 같이 움직이나 | from-scratch GDN-2 | 부하↑ 시 recall 붕괴하는데 eRank는 계속 상승 |
| E5 | `linear-memory-routing/report/0022.md` | write를 펴면(decorrelate) recall이 오르나 (decay 없는 조건) | from-scratch delta-net | 과부하에서만 소폭 이득, 여유/반포화에선 중립~손해 |
| E6 | `linear-memory-routing/report/0023.md` | 위 개입이 실제 GDN-2(decay)에서도 되나 | gdn2-1.3B(fla, 학습) | 전 조건 실패(recall↓, eRank 붕괴) |
| E7 | `rebuild/nb6_S21_chunking.py` | eRank 포화를 경계 신호로 쓰면 고정분할과 다르게 찍히나 | gdn2-1.3B | 약한 상관(ρ=−0.38, p=0.097) — 미결 |

모델 참고: **gdn2-1.3B-fineweb-edu (checkpoint-10B)**, lit_gpt `Config.from_name("gdn2_1.3B")`,
18 layer, head_dim 128. from-scratch 실험은 별도 소형 모델(아래 각 항목 세팅 참조).

---

## 3. 실험별 상세

### E0 — 대수: additive multi-state는 delta의 erasure를 위반 (S8/F3)
- **목적**: state를 여러 조각으로 나눠 보관할 때, "나중 기록이 이전 기록을 덮어쓴다"는 delta rule의
  기능이 스냅샷 합치기에서 살아남는지 확인.
- **세팅**: numpy 수식 toy(모델 없음). d=32, delta rule β=1, key L2-norm. 같은 key에 A 저장 후 B로
  갱신. filler key-value 쌍 포함. 세 읽기 방식 비교 — (a) 단일 연속 state, (b) 스냅샷 additive read
  (스칼라 가중), (c) 연산자-합성 merge(S¹에 P=(I−kkᵀ) 적용 후 합침).
- **결과(보수적)**: (a) A=0.00, B=1.00, filler=1.00 (정상). (b) 동일 가중 additive는 A=1.00
  (지운 값 부활), B=1.00, filler=1.00. 스칼라로 A를 죽이면(w1→0) A=0.00이지만 filler=0.00(같이 죽음).
  (c) 연산자-합성 merge만 A=0.00, B=1.00, filler=1.00.
- **함의**: 스냅샷을 스칼라로 더해 합치면 지운 값을 원리적으로 제거 못 한다. 행렬 연산자로 합쳐야 함.
- **한계**: 수식 toy. 실제 모델·데이터가 아니라 대수적 예시.

### E1 — eRank vs 부하, decay vs no-decay (S12+S13)
- **목적**: (RQ1) eRank가 부하 따라 오르나, (RQ2 일부) decay가 eRank를 누르나 — **통제된 대조**로.
- **세팅**: from-scratch 2-layer 소형 모델 두 개, 동일 head_dim=32·동일 width·동일 MQAR·동일 학습(5000
  step). 하나는 **GatedDeltaNet2(decay 있음)**, 하나는 **DeltaNet(decay 없음)**. **시퀀스 길이 T=384로
  고정**(부하 kv를 4→96으로 바꿔도 T 불변 → eRank 상승이 길이가 아니라 부하 때문임을 보장). eRank는
  최종 state의 특이값 엔트로피, head 평균.
- **결과(보수적)**:
  - 두 모델 모두 부하↑ → eRank↑ (DeltaNet 9.3→22.4, GDN-2 6.7→11.2, kv 4→96). → RQ1 성립.
  - 모든 부하에서 **no-decay(DeltaNet) eRank가 decay(GDN-2)보다 약 2배 높음** (예 kv16: 17.0 vs 9.5).
    recall도 no-decay가 높음(kv96: 0.52 vs 0.22).
- **함의**: 같은 용량이면 **decay가 eRank(와 recall)를 낮춘다**.
- **한계**: 소형 2-layer, MQAR로만 학습한 task-specific 모델, 단일 seed. 절대 수치는 실모델을 대표하지
  않음(메커니즘 방향만 신뢰). 저부하(kv 작음)에선 T=384의 대부분이 inert 패딩 토큰(id 0)임.

### E2 — 데이터 유형별 eRank 궤적 (S14)
- **목적**: 자연어/수학/코드/지식 등 데이터 유형에 따라 eRank(위치별) 궤적이 갈리는지.
- **세팅**: gdn2-1.3B. 유형별 4개 시퀀스(길이 256), 위치별 eRank(layer [4,8,12] 평균)를 **평균±표준편차
  밴드**로. (이전 버전은 단일 시퀀스여서 밴드 없이 갈려 보였음 — 이번에 다중 시퀀스로 교정.)
- **결과(보수적)**: 네 유형의 궤적이 상승 후 포화(최종 eRank 13.6~15.4). **밴드가 크게 겹쳐 유형 간
  분리는 약함.** 단일 시퀀스로 볼 때의 "유형별 분리"는 대체로 노이즈였음.
- **함의**: eRank는 **다양 vs 반복**은 크게 가르지만(별도 관찰), **서로 다른 다양한 유형끼리는 잘
  구별하지 못함**.
- **한계**: 유형별 4 시퀀스, 단일 모델. layer 3개 평균.

### E3 — 낮은 eRank의 원인 분해: decay vs 뭉침 (S15/F6, S16/F7)
- **목적**: 실모델의 낮은 eRank가 **decay(시간에 따른 감쇠)** 때문인지 **key 방향 뭉침(비등방)**
  때문인지 반사실적으로 분해.
- **세팅**: gdn2-1.3B, 자연어 입력, layer [2,5,8,10,13,16]. forward에서 커널 입력(q,k,v,decay g)을
  로그한 뒤 state를 2×2로 재조립 — {실제 decay, decay 끔(g=0)} × {실제 key, 등방 랜덤 key}.
  **등방 key는 실제 key와 per-vector norm을 맞춤**(방향만 랜덤화; 크기 교란 제거). value는 고정.
  eRank cap = head_dim 128.
- **결과(보수적, layer 평균)**: 실제 state eRank = **20.1 / 128** (약 84%가 유휴). decay를 끄면
  **+43.7**(→63.8), key를 등방으로 하면 **+9.9**(→30.0), 둘 다 하면 63.9. 상호작용항 −9.8(약간
  sub-additive). → **decay 몫이 뭉침 몫의 약 4.4배.**
  - F6(head별 decay r̄ vs eRank 산점도): head별 eRank 3.4~56.5로 넓게 분포, 이론곡선 e/(1−r̄)의
    상승 추세를 대체로 따름.
- **함의**: 낮은 eRank의 **주원인은 decay**, key 뭉침은 실재하나 **부차적**.
- **한계**: 단일 코퍼스 표본, 단일 forward(단일 seed). F6은 상관 관계 그림이며, 일부 head 점이
  이론곡선 위로도 흩어져 "곡선 아래=뭉침 몫"이라는 읽기는 약함(인과 분해는 F7이 담당). "84% 유휴"는
  이 입력·이 모델 기준.

### E4 — eRank는 용량인가 계기판인가 (S18)
- **목적**: eRank가 높으면 recall이 좋은가? 둘을 **같은 조건에서 짝지어** 측정.
- **세팅**: from-scratch GDN-2(head_dim 32) MQAR 학습 후, 부하 kv를 2→96으로 sweep하며 **recall과 최종
  state eRank를 동시에** 측정.
- **결과(보수적)**: 부하↑ 시 **recall은 1.00→0.13으로 붕괴**하는데 **eRank는 4.9→17.7로 계속 상승·
  포화**한다. 둘은 같이 안 움직이고 오히려 벌어진다.
- **함의**: **eRank ≠ 용량.** 높은 eRank가 좋은 recall을 뜻하지 않는다. eRank는 부하를 반영하는
  **계기판**이지, 올리면 성능이 오르는 손잡이가 아니다.
- **한계**: from-scratch 소형 모델, 단일 seed. 이 sweep은 kv에 맞춰 시퀀스 길이도 함께 커짐(T 고정
  아님) — 단 "recall 붕괴 vs eRank 상승"이라는 발산 관찰 자체는 길이 confound와 무관하게 성립.

### E5 — write를 펴면 recall이 오르나 (decay 없는 조건) (0022)
- **목적**: 유휴 차원을 쓰도록 **write를 decorrelate/직교화**(설계 아이디어 A)하면 multi-key recall이
  오르는지. E3에서 뭉침은 부차 요인이지만, "그래도 뭉침을 풀면?"을 직접 시험.
- **세팅**: from-scratch delta-net(**decay 없음** — 뭉침 요인만 고립). 3개 용량 regime(여유 H2×64 /
  반포화 H1×32 / 과부하 H1×16). 후보 3종: C4(key Gram 벌점=decorrelation 정규화), C1(key whitening),
  C2(학습 linear map). 모두 write key와 read query에 동일 적용(검색 보존).
- **결과(보수적)**:
  - 여유(recall≈1): 개입 중립.
  - 반포화(baseline kv64=0.961): 개입이 **손해**(C1 kv64 0.961→0.765).
  - 과부하(baseline kv64=0.081): 개입이 **소폭 이득** — C4 0.081→0.181, C1 0.081→0.153. C2는 무효.
- **함의**: decay 없는 조건에서 (A)는 **진짜 과부하일 때만** 값을 한다. C2(자유 학습 map)는 명시적
  압력이 없어 효과 없음.
- **한계**: decay 없는 toy, 단일 seed. 절대 수치 작음.

### E6 — 같은 개입이 실제 GDN-2(decay)에서도 되나 (0023)
- **목적**: 0022의 승자 C4(정규화)를 **실제 decay·erase gate가 있는 GDN-2**에 이식해 검증.
- **세팅**: from-scratch로 fla GatedDeltaNet2(실제 gating) 학습(MQAR). C4를 k_proj 출력에 Gram 벌점으로
  얹음(커널 무수정). 3 config: hd32·λ0.5 / hd32·λ0.1 / hd16·λ0.5. baseline vs C4.
- **결과(보수적)**: **전 조건에서 C4가 recall을 낮춤** (hd32: 0.497→0.430; λ 낮춰도 0.497→0.480;
  hd16: 0.110→0.045). 동시에 state eRank가 오르기는커녕 **붕괴**(16.2→6.2 및 2.8).
- **함의**: **decay가 있는 실모델에선 (A)가 전이되지 않고 오히려 해롭다.** E3대로 실모델 병목은
  decay이므로, key만 억지로 펴면 학습된 key↔gate 관계와 충돌한다.
- **한계**: from-scratch 학습(사전학습 대형모델 미검증), C4만 이식(C1/C2는 미이식), 단일 seed.

### E7 — eRank 포화를 chunking 경계 신호로 (S21)
- **목적**: eRank가 포화하는 지점을 "여기서 state를 끊자"는 경계로 쓰면, 고정분할과 다르게 정보
  밀도를 따라 찍히는지.
- **세팅**: gdn2-1.3B. 반복률(unique_frac)로 정보 밀도를 조절한 시퀀스에서 eRank-포화 기준으로 청킹,
  **고정분할 baseline**(밀도 무관 일정 길이, 예산 동일)과 비교. 밀도=예측 엔트로피(bits/token).
- **결과(보수적)**: 적응형 청크 길이 vs 밀도 상관 = **Spearman ρ=−0.38, p=0.097 (n=20)** — 방향은
  맞으나(밀도↑→청크 짧아짐) **약하고 통계적으로 유의하지 않음**. 밀도 지표가 반복열에서 반직관적으로
  움직이는 문제도 있음.
- **함의**: eRank를 경계 신호로 쓰는 건 **아직 미결**. 고정 baseline은 추가됨(예전 그림엔 없었음).
- **한계**: 단일 모델, n=20, 밀도 지표 신뢰도 의심. 확정적 결론 없음.

---

## 4. 객관적 사실 정리 (방어 가능한 것만)

1. **eRank는 부하를 따라간다.** 외울 kv 쌍이 늘면 최종 state eRank가 단조 증가한다 (E1, E4).
2. **낮은 eRank의 주원인은 decay다.** 실모델(1.3B) 반사실 분해에서 decay 제거 시 eRank +43.7,
   key 등방화 시 +9.9 — decay 몫이 약 4.4배 (E3). 통제된 from-scratch 대조에서도 no-decay 모델의
   eRank가 decay 모델의 약 2배 (E1). **key 뭉침은 실재하나 부차적.**
3. **eRank ≠ 용량(recall).** 부하가 오르면 eRank는 계속 상승하는데 recall은 붕괴한다 (E4). eRank를
   write 기하로 올려도 실모델에선 recall이 오르지 않고 오히려 떨어진다 (E6). → eRank는 **읽는 계기판**
   이지 **올리는 손잡이가 아니다.**
4. **설계 아이디어 (A)(write decorrelation)는 실모델에서 실패한다.** decay 없는 toy의 과부하에서만
   소폭 이득(E5), decay 있는 실제 GDN-2에선 전 조건 손해 + eRank 붕괴(E6). → (A) 계열 폐기, 설계
   레버는 decay/retention 쪽.
5. **multi-state를 스칼라로 더해 합치면 delta의 erasure가 깨진다** (E0, 대수 toy). 지운 값을 지우려면
   행렬 연산자 합성이 필요.
6. **eRank의 데이터-유형 변별력은 약하다.** 다양 vs 반복은 가르지만, 서로 다른 다양한 유형끼리는 궤적이
   겹친다 (E2).
7. **eRank를 chunking 경계 신호로 쓰는 것은 아직 검증되지 않았다** (E7, ρ 약함·비유의).

---

## 5. 아직 모르는 것 / 열린 문제

- **decay/retention을 직접 다루는 설계**(예: 중요한 방향의 감쇠를 늦추는 selective retention)가
  recall을 올리는지 — 미검증(다음 방향).
- eRank(또는 다른 신호)를 경계 신호로 쓰는 dynamic chunking이 고정분할 대비 실이득이 있는지 — E7은
  미결.
- 위 모든 것의 **사전학습 대형모델·자연어 벤치마크 전이** — 대부분 소형/합성/단일 seed에서만 봄.

## 6. 전반 한계 (모든 결과에 공통)

- 다수 실험이 **from-scratch 소형 모델 + 합성 MQAR + 단일 seed**다. 방향성 증거이지 최종 수치가 아니다.
- 실모델 실험(E2/E3/E7)은 gdn2-1.3B **단일 체크포인트·단일/소수 표본**이다.
- MQAR 저부하 구간은 대부분 inert 패딩으로 채워진다(자연 문맥과 다름).
- mamba-ssm 빌드가 현재 환경(torch 2.13+cu130)에서 막혀, mamba2 계열은 이번 정리에서 제외했다.

---

## 부록 — 코드·리포트 위치
- 그림 재작성: `sohyung's_brain/rebuild/` (nb1_F3, nb2_S12_S13, nb2c_S14, nb3_F6F7, nb4_S18, nb6_S21) +
  각 PNG. 상태·감사: `rebuild/README.md`, `labmeeting_figure_audit.md`, 발표 구조: `labmeeting_refactor.md`.
- 설계 개입: `linear-memory-routing/report/0022.md`, `0023.md` (+ `report/0022_code/`,
  `report/0023_code_gdn2_c4.py`).
- 실행 환경 복구: `rebuild/VESSL_RECOVERY.md` (`/root/smaller/sh_rebuild/sh_setup.sh`).
