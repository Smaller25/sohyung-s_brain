# Lab-meeting figure 세팅 감사 (2026-07-15)

`labmeeting_refactor.md`의 슬라이드별 **세팅 체크리스트**와, 두 폴더에 실제로 넣어둔 figure들의
**생성 코드**를 대조한 결과. (memory routing = `linear-memory-routing/report/labmeeting_figures/`,
SSM state analysis = `SSM_Rank_Analysis/notebooks/`.) 코드 정밀 조사 + PNG 육안 확인 병행.

## 한눈에 — 판정 요약
| 슬라이드 | figure | 폴더 | 판정 | 핵심 문제 |
|---|---|---|---|---|
| S8 | F3_delta_toy | routing | 🟡 부분 | 조건(c) 연산자-merge 막대 없음, filler·최적λ 없음 → 약한 버전 |
| S12 | worked_example_S1_D1_both | SSM | 🟡 부분 | **T가 kv에 따라 증가**(T=3·kv) → erank↑가 길이 탓일 여지 (체크리스트는 T 고정 요구) |
| S13 | decay_mqar | SSM | 🟢 일치 | — |
| S14 | signal_trajectories_raw | SSM | 🟡 부분 | **단일 시퀀스** 궤적 (±밴드·다시퀀스 평균 아님) |
| S15 | F6_erank_vs_decay | routing | 🟡 부분 | **repetitive 입력**(자연어 아님), 대본 "64칸"→실제 cap 128 |
| S16 | F7_decomposition | routing | 🟡 부분 | **등방 key norm 미매칭**(randn_like로 크기까지 랜덤=교란), repetitive 입력 |
| S17 | F8_concentration | routing | 🟢 일치 | (repetitive 입력이나 체크리스트 항목엔 영향 없음) |
| S18 | F9_datatypes | routing | 🔴 **불일치** | **recall 축이 없음** — S18 주장("erank–recall regime")을 못 받침. figure 배정 오류 |
| S21 | chunk_by_density | SSM | 🟡 부분 | **고정분할 baseline 없음**(동일 budget 통제 X), mamba2만 |
| S22 | worked_example_boundaries | SSM | 🔴 **불일치** | 텍스트·ground-truth 전환점 없음, 예시 2개(seed 0)뿐, 정렬률 수치 없음; "실제 시퀀스"가 합성 반복열 |

🟢 일치 3 · 🟡 부분 5 · 🔴 불일치 2.

---

## 🔴 발표를 막는 2건 (우선 처리)

### S18 / F9 — figure가 주장과 다름 (제일 중요)
- **S18 주장**: "erank와 recall은 과적재일 때만 함께 움직인다 → eRank≠capacity (계기판)."
- **실제 F9_datatypes.png**: `erank by input type × intervention` (자연어/math/code/knowledge/repetitive
  × real/g=1/iso_k/iso_v/all_off). **recall을 아예 측정/표시하지 않음.** 내용상 F7을 데이터유형별로
  확장한 그림이지, erank–recall 관계 그림이 아님. MANIFEST도 이걸 "입력-의존성"으로 정확히 라벨.
- **처리**: S18에는 **erank↑인데 recall↓**를 보이는 그림이 필요. 후보:
  (1) `candidates/F6_F9_load_vs_horizon.png` (SSM_Rank: eRank↑ vs recall↓ anti-correlation),
  (2) **0022/0023의 regime별 erank↔recall** (R1 여유=무관, R3 과적재=동반; 0023 real GDN=erank 붕괴).
  → S18을 0022/0023과 통합하거나(그 편이 P3 서사와 자연스러움), load_vs_horizon로 교체.
  현재의 F9_datatypes는 **S16(F7)의 백업/보강**으로 돌리는 게 맞음.

### S22 / worked_example_boundaries — 체크리스트 4항 전부 미충족
- **실제**: eRank(t) 곡선 + 경계 세로선 2개만. **텍스트 없음, topic 전환점/needle 위치 마커 없음**,
  예시 2개(unique_frac 0.1/1.0, seed 0)뿐, **정렬률 수치 없음.** 게다가 "시퀀스"가 WikiText 창을
  타일링한 **합성 반복열**이라 진짜 의미 전환이 없음 → 경계-의미 정렬을 테스트하는 그림이 아님.
- **처리**: [진행 중] 딱지엔 맞지만 이 상태로 S22에 쓰면 "경계가 의미와 정렬" 주장을 못 함.
  자연어 실텍스트 위에 경계 + ground-truth(topic/needle) 마커 + "무작위 N개 중 정렬률" 필요.
  당장 없으면 S22는 "계획(S23)"과 합치거나 "예시 곡선(정렬 검증은 계획)"으로 톤다운.

---

## 🟡 부분 일치 — 고치면 되는 것

### S16 / F7 — 등방 key norm 미매칭 (교란변수)
- 코드: `if rk: kw2['k']=torch.randn_like(kw['k'])` → 방향+**크기**를 모두 랜덤화.
  체크리스트는 "방향만 랜덤, norm은 실제 key와 매칭"을 명시(크기 바꾸면 교란).
- 영향: decay +7.8 / 뭉침 +1.5 결론의 **방향**은 유지될 개연성이 크나, 뭉침 몫(+1.5) 크기가
  norm 차이에 오염됐을 수 있음. **재생성 권장**: `k_iso = randn; k_iso *= k.norm(dim=-1,keepdim=True)/k_iso.norm(...)`.
- 부차: 상호작용항(≈−0.8, sub-additive)은 계산 가능하나 그림에 라벨 없음 → 캡션 한 줄 추가.

### S15 / F6 — repetitive 입력 + 대본 숫자 오류
- 입력이 `'The history of science...'×80` (degenerate). MANIFEST 스스로 "F6/F8은 repetitive라
  erank 과소평가"라고 적어둠. 그래도 **per-head decay↔erank 곡선의 형태**는 입력에 비교적 robust.
- 대본 S15 "최대 64칸인데 10~17칸" → 실제 cap **128**(GDN2-370m head_dim). "64"를 128로 정정.
- **처리**: 이상적으론 자연어(F9의 natural) 입력으로 F6 재생성. 급하면 "repetitive 입력" 캡션 명시.

### S8 / F3 — 약한 버전 (조건 (c)·filler·최적λ 없음)
- 충족: β=1, key L2-norm, v_A⊥v_B, A/B 스택 bar (single A=0/additive A=1).
- 미충족: **(c) 연산자-합성 merge 막대 없음** — "제대로 합치면 A≈0으로 복귀"라는 더 강한 대비가 빠짐.
  **filler 쌍 없음** + additive가 **동일 가중**(최적 λ 아님) → 비판자가 "additive에 최선 기회를 안 줬다"
  고 할 수 있음. 체크리스트는 filler+최적λ에서도 additive가 실패함을 보이라고 요구.
- **처리**: 막대 3개(single/additive-optimalλ/operator-merge) + filler 포함으로 재생성하면 S8 논증이 방탄.

### S14 / signal_trajectories_raw — 단일 시퀀스
- `ids1 = ids[:1]`로 첫 시퀀스만. 데이터유형별 **다시퀀스 평균 ±밴드 아님** → "일화(anecdote)" 위험.
  T는 D1/D2/D3 모두 192로 맞음(OK). head-mean은 코드상 맞음(캡션 명시 권장).
- **처리**: n_seq(생성은 4개 있음)로 평균+밴드 재플롯.

### S12 / worked_example_S1_D1_both — T가 고정 아님
- `T=3·n_pairs`로 kv가 늘면 시퀀스도 길어짐. 체크리스트 1항("T 고정, filler로 길이 맞춤")과 배치.
  erank↑가 load 때문인지 길이 때문인지 **분리 안 됨.** (나머지 항목 3개는 충족: 같은 ckpt, entropy 정의,
  축 x=kv/y=erank±std.)
- **처리**: 최대 kv 길이로 padding해 T 고정 후 재측정하면 clean. 급하면 "T가 load에 비례 증가" 캡션 명시.

### S21 / chunk_by_density — 고정분할 baseline 없음
- eRank-포화 vs epiplexity 두 **적응형** 기준만 비교, **고정분할(동일 chunk 수) baseline 없음**.
  → "적응 경계가 고정 대비 낫다"를 못 보임(밀도 효과와 포화 효과 분리 안 됨). mamba2-370m만.
- **처리**: 동일 budget 고정분할 baseline 추가.

---

## 🟢 일치
- **S13 / decay_mqar**: 6개 pretrained 모델 동일 MQAR 평가, decay(dashed)/no-decay-deltanet(solid) 구분,
  erank/max_rank 정규화로 스케일 비교 가능. 체크리스트 3항 충족.
- **S17 / F8**: head별 key 코사인 히스토그램 + 등방 null 오버레이(4000쌍/head). 2항 충족. (repetitive
  입력이나 이 그림 주장에는 무관.)
- **S12**: 위 T 주의만 빼면 정의·축·통제 양호.

---

## 관통 이슈 (여러 figure 공통)
1. **repetitive 입력 재사용** (F6/F7/F8): MANIFEST가 이미 인지. 발표 정본은 자연어/실데이터로.
   F9(자연어 포함)가 이미 있으니 F6/F7도 같은 입력으로 재생성하면 일관성↑.
2. **합성/반복열이 "실데이터"로 보일 위험** (S22, chunking): unique_frac 반복열은 density 통제엔 좋으나
   "의미 전환 정렬"은 못 보임. 자연어 topic-shift/needle이 필요한 주장과 구분해 캡션.
3. **단일 시퀀스/single seed** (S14, S22, 그리고 0022/0023): 캐비앗 각주 필수(발표 문서도 이미 명시).
