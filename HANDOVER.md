# 인수인계 — Make_kirotoo (2026-11-28 KST)

## 1. 하던 일
v20 다리 · v21 몸통 · 머리를 실물 출력 · 조립에서 부러지지 않게 고치는 중(소유자 「이음 · 걸쇠가 다 부러짐, 해상도 낮다고 가정」). 2차 수정 끝: 다리 `print_robust`(0bb03c7, STL 030012b — 발판 혼 판 보강, 발목 롤 나사 8개, 터널 살 3, 눈물방울), 몸통 · 머리(5f3ba33 · 346086a, STL ec00e1c — 껍데기 벽 2.0, 도브테일 4.5 · 목 7 · 틈 0.5, 판 220 안). 팔꿈치 팔(팔 모터 4개) 채택 → 같은 내구 기준 적용을 서브에이전트에 맡김(build21_arm.py, 진행 중).

## 2. 다음 한 걸음
- 팔 내구 보강 끝(STL 271132f) — 출력 묶음 `out20/PRINT_v2/` 4 폴더(STL 은 하드링크라 원본 STL 을 다시 만들면 `ln -f` 로 다시 이어야 함, 그림 · 점검표는 `print_tests/tools/stl_png.py` · `print_sheet.py out20/PRINT_v2` 로 다시). 팔 문서 5–8절 옛 숫자 · `arm21.py sweep post` 다시 돌리기 남음.
- ~~그다음 한 묶음: MJCF 다시 내보내기~~ ✓ v22 aa0c1cd. 남은 설계 쪽: 팔 0° 에서 고관절 롤 바깥 −7.2° 막힘(걷기 팔 자세 어깨 −15° / −25° 결정과 함께 볼 것), 상자 크기는 `sim/tests/contact_v22.py` · `shape_contact_v22.py` 로 형상 바뀌면 다시. 옛 줄: — 2차 형상 질량(머리 +130 g, 전체 +200 g, 팔), 다리 ↔ 몸통 · 팔 ↔ 다리 · 머리 ↔ 다리 충돌 쌍(지금 v21_rl 에 없음, DGX_2 실측 — 몸통 상자 아래 면은 −115° 에서 3 mm 겹침, 실물 반바지 치수와 대조해 맞출 것), 관절 한계를 형상값으로(고관절 피치 −117.8 / +113.9, 무릎 117.8 → 여유 둔 115) → `sim/export_mjcf_rl.py` · `check_v21.py` → kitoru-rl `assets/` 갱신. 같은 묶음에서 끄덕임 크랭크 서보 토크 다시(머리 무거워짐, `code_v17/out_head21/HEAD_CRANK_v21.md`).
- 현황판 10번 「남음」 ①–⑥: 발판 혼 나사 자리(채움 100 % 권고 7c28e7f, 금 가면 혼 판 분리) · 반바지 벽 1.2 · Jetson B01 · 범위 sweep · 조립도 · 옛 이음 필렛.

## 3. 도는 것
- 조립도 v21 끝(8f59659, A3 4장). 서브에이전트(11-28): `shorts.stp` 플랜지 구멍이 별도 솔리드로 더해져 통판(STL 100 cm³ vs 설계 30.9) — 고친 뒤 메인이 shorts STL · PNG · PRINT_v2 `ln -f` · `print_sheet.py` · 조립도 다시(`ASM_ONLY`).
- 없음. 서포트 없애기 끝(220db8f · 8dd1d5e), STL 4개 · PRINT_v2 갱신. 남은 것: build20 `__main__` 산출물(`build20_log.json` · `biped_robot_v20.stp` · `model.json`)은 서포트 없애기 전 판 — 조립도 다시 만들 때 같이.
- 없음. v22 모델 끝(aa0c1cd · kitoru-rl 8c1c728), DGX_2 에 넘김 — 그쪽 할 일: Warp 불러 보기, 일어나기 시작 자세 다시(17.8 % 겹침), 비트에 기댄 스크립트 둘.
- DGX_2 RC 세션 「DGX_2 설치 작업 지시」: kitoru-rl 갈래 넷을 스스로 계속 돌림(소유자 「끝」까지). 메인에는 좋은 결과(영상) · 설계 결정만 보냄 → 영상은 `project_media_put(project="kitoru-rl")` + SendUserFile.

## 4. 감시와 기다림
- git 감시 없음(소유자: RC 세션이 직접 알림). 소유자 답 대기: Jetson B01 STEP(로그인 필요) 또는 실측, 발판 혼 판 방식(A 채움 100 % 권고).

## 5. 소유자 말 (이번 세션 끝 무렵)
- 「1,2,3 다 진행, 덮개는 나사 말고 끼우는 걸 키워」 → 도브테일 4.5 · 7 · 0.5.
- 「시드는 최대 2개」(모든 프로젝트, 비서가 공통 규칙으로 넘김) · 「좋은 결과는 영상, 끝이라 할 때까지 쉬지 말고, 개선 적용하며」(kitoru-rl CLAUDE.md) · 30분 시험에서 나은 판은 카드 1장 2시간까지 미리 허락.
- 강화학습 규칙 순서 ① 잘 걷기 ② 안 넘어짐 ③ 넘어져도 일어남 ④ 빠르게 ⑤ 누워서라도 이동 ⑥ 목표 지점(kitoru-rl 중기 목표).
- 「머리 무게 일단 냅둠」 · 팔은 팔꿈치 팔 4모터 · 「DGX2 는 알아서, 너는 설계에 집중」.
- Jetson 은 Nano 4GB 개발 키트 B01(사진 `references/images/images.jpg`) — 2GB 아님: 원통 잭 5 V 4 A, 강압 5 V 3 A 로 모자람, 팬을 방열판 위로.

## 6. 열린 것
- 소유자: Jetson B01 STEP · 실측, 발판 혼 판(A 채움 100 % / B 분리 / C 눕힘), 반바지 벽(범위 sweep 뒤).
- 무게 늘어남(+200 g)이 걷기 · 끄덕임 서보에 주는 영향 — 다음 모델 판에서 계산.

## 7. 건드리면 안 되는 것
- `robot/legacy/XL430_robot_base.py`(원본), `references/`(git 제외 원본 STEP · 사진).
- 전체 `export_parts.py build20`(몸통 · 머리 공차를 되돌림) — 다리는 3번째 인자로 조각만.

## 8. 글에 없는 것
- 소유자는 시간이 없다 — 짧게, 출력 가능 여부부터. 선택지를 말하면 A/B 가 무엇인지 반드시 풀어 말한다.
- 받은 목표 · 추천에 시드 3 이상이 있으면 2 로 고쳐 넘긴다(소유자가 크게 화냄).
- 서브에이전트는 코드 + 빌드 + STEP 만, STL · PNG 는 메인이 병렬(16개 약 20초–1분).
- 관제실 허브 kitoru-rl · Make_kirotoo 생김(11-27) — gif 는 1 MB 넘으면 거부, 영상은 mp4.

## 9. 읽을 곳
`STATUS.md` → `CLAUDE.md` → `out20/LEG_PRINT_KIT_R.md`(맨 위 두 절) → `out20/BODY_HEAD_PRINT_KIT.md` 0.0 절 → `out21_arm/ARM_ELBOW_v21.md` → `sim/getup/GETUP_STUDY.md` → kitoru-rl `STATUS.md` · `CLAUDE.md`
