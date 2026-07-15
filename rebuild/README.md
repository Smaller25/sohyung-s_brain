# Figure rebuild — 최소 티어 (2026-07-15)

`labmeeting_figure_audit.md`에서 나온 문제를 고쳐 다시 만든 figure들. 최소 티어 = 발표 신뢰도에
직결되는 것만 (전체 6-노트북 재작성 아님). 각 항목은 만들 때마다 개별 커밋·푸시.

## 진행 상태
| 슬라이드 | 항목 | 상태 | 산출물 |
|---|---|---|---|
| S8 | F3 대수 toy (조건c·filler·최적λ) | ✅ done | `nb1_F3_algebra.py`, `F3_delta_toy_fixed.png` |
| S18 | erank↔recall sweep (F9 대체) | ✅ done | `nb4_S18_erank_gauge.py`, `S18_erank_vs_recall.png` |
| S16 | F7 분해 (norm-matched iso-k, 자연어, 1.3B) | ✅ done | `nb3_F6F7_decomposition_1p3B.py`, `F7_1p3B.png` |
| S15 | F6 decay↔erank (자연어, 1.3B, cap 정정) | ✅ done | `nb3_F6F7_decomposition_1p3B.py`, `F6_1p3B.png` |
| S12 | eRank↑ with load (FIXED T) | ✅ done | `nb2_S12_S13_decay_vs_nodecay.py`, `nb2_S12_S13.png` |
| S13 | decay suppresses eRank (from-scratch) | ✅ done | 동상 (DeltaNet no-decay ≫ GDN-2 decay) |
| S14 | 데이터유형별 erank 궤적 (multi-seq band) | ✅ done | `nb2c_S14_trajectories.py`, `nb2c_S14.png` — **밴드 겹침: 유형간 분리 약함(정직)** |
| S22 | worked_example_boundaries | 🗑️ **삭제 결정** | 아래 참고 |

## S22 — 본문에서 삭제
감사 결과 체크리스트 4항 전부 미충족(텍스트·GT 전환점 없음, 예시 2개, 정렬률 없음, 합성 반복열).
현 상태로는 "경계가 의미와 정렬"을 못 보임 → **본문에서 제거**. "경계-의미 정렬 검증"은
S23(RQ4 검증 계획)으로 흡수. S21(chunk_by_density)이 이미 [진행중] 신호 증거를 지니므로
S22를 빼도 RQ3 서사에 구멍 없음. (자연어 실텍스트+topic/needle GT+정렬률로 제대로 만들려면
새 실험 = "다음 학기" 범위.)

## 모델
- 실모델 figure(F6/F7)는 이제 **gdn2-1.3B-fineweb-edu-100b (checkpoint-10B)** 사용 (기존 370m 대체).
  로딩: dscpkg `lit_gpt` `Config.from_name("gdn2_1.3B")` + `load_state_dict(strict=False)` (missing/unexpected 0).
- 실행 환경: VESSL A100, torch 2.13+cu130, vendored fla 0.5.2 (`/root/vfla`).

## 주요 수정 (감사 대비)
- **F3**: filler 쌍 + additive 최적 스칼라(A 죽이면 filler도 죽음) + 조건(c) 연산자-merge P=(I−kkᵀ) → A≈0·filler보존.
- **S18**: recall 없는 F9_datatypes 대신, load sweep에서 recall↔erank 짝측정 (erank↑ while recall↓).
- **F7**: iso-key를 `randn_like`(norm까지 랜덤=교란) → **norm-matched**(방향만 랜덤)로 교정 + 자연어 입력 + 상호작용항 표기.
- **F6**: repetitive 입력 → 자연어, 이론곡선 cap을 실제 head_k_dim으로.
