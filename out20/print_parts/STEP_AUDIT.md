# STEP 솔리드 점검

`print_tests/tools/step_audit.py` 가 쓴 표(gmsh/OpenCASCADE 로 STEP 을 읽어 잼). 기준: brep = gmsh 솔리드 수, 음의 부피 0, 표면 메시 열린 모서리 0(step2stl.py 와 같은 옵션), 겹침(솔리드 합 − 합친 부피) ≤ 15 %(설계 솔리드는 보스 · 덧판 · 0.05 이음으로 원래 겹침), parts.json 부피와 솔리드 합 차 ≤ max(0.06 cm³, 1 %), 합친 뒤 한 덩어리.

다시 재기: `LD_LIBRARY_PATH=sim/tmp/glu uv run -q --no-project --python 3.12 --with gmsh --with numpy python -I print_tests/tools/step_audit.py out20/print_parts out21_arm/print_parts --md out20/print_parts/STEP_AUDIT.md`
| 부품 | brep | gmsh | 음의 부피 | 열린 모서리 | 솔리드 합 cm3 | 합친 부피 cm3 | 겹침 cm3 | parts.json cm3 | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| `out20/print_parts/L_ankle_bracket.stp` | 25 | 25 | 0 | 0 | 35.84 | 33.26 | 2.58 | 35.8 | 깨끗 |
| `out20/print_parts/L_foot_sole.stp` | 22 | 22 | 0 | 0 | 70.65 | 67.48 | 3.17 | 70.6 | 깨끗 |
| `out20/print_parts/L_hip_roll_bracket.stp` | 13 | 13 | 0 | 0 | 32.26 | 30.40 | 1.86 | 32.3 | 깨끗 |
| `out20/print_parts/L_hip_yaw_bracket.stp` | 9 | 9 | 0 | 0 | 23.61 | 22.73 | 0.88 | 23.6 | 깨끗 |
| `out20/print_parts/L_shank_frame.stp` | 33 | 33 | 0 | 0 | 35.70 | 34.02 | 1.69 | 35.7 | 깨끗 |
| `out20/print_parts/L_shank_shell.stp` | 8 | 8 | 0 | 0 | 15.62 | 15.24 | 0.38 | 15.6 | 깨끗 |
| `out20/print_parts/L_thigh_frame.stp` | 27 | 27 | 0 | 0 | 40.44 | 36.97 | 3.47 | 40.4 | 깨끗 |
| `out20/print_parts/L_thigh_shell.stp` | 10 | 10 | 0 | 0 | 5.62 | 5.48 | 0.14 | 5.6 | 깨끗 |
| `out20/print_parts/R_ankle_bracket.stp` | 25 | 25 | 0 | 0 | 35.84 | 33.26 | 2.58 | 35.8 | 깨끗 |
| `out20/print_parts/R_foot_sole.stp` | 22 | 22 | 0 | 0 | 70.65 | 67.48 | 3.17 | 70.6 | 깨끗 |
| `out20/print_parts/R_hip_roll_bracket.stp` | 13 | 13 | 0 | 0 | 32.26 | 30.40 | 1.86 | 32.3 | 깨끗 |
| `out20/print_parts/R_hip_yaw_bracket.stp` | 9 | 9 | 0 | 0 | 23.61 | 22.73 | 0.88 | 23.6 | 깨끗 |
| `out20/print_parts/R_shank_frame.stp` | 33 | 33 | 0 | 0 | 35.70 | 34.02 | 1.69 | 35.7 | 깨끗 |
| `out20/print_parts/R_shank_shell.stp` | 8 | 8 | 0 | 0 | 15.62 | 15.24 | 0.38 | 15.6 | 깨끗 |
| `out20/print_parts/R_thigh_frame.stp` | 27 | 27 | 0 | 0 | 40.44 | 36.97 | 3.47 | 40.4 | 깨끗 |
| `out20/print_parts/R_thigh_shell.stp` | 10 | 10 | 0 | 0 | 5.62 | 5.48 | 0.14 | 5.6 | 깨끗 |
| `out20/print_parts/antenna_L.stp` | 1 | 1 | 0 | 0 | 2.86 | 2.86 | 0.00 | 2.9 | 깨끗 |
| `out20/print_parts/antenna_R.stp` | 1 | 1 | 0 | 0 | 2.86 | 2.86 | 0.00 | 2.9 | 깨끗 |
| `out20/print_parts/arm_L_finger_rod.stp` | 2 | 2 | 0 | 0 | 1.73 | 1.73 | 0.01 | 1.7 | 깨끗 |
| `out20/print_parts/arm_L_flap.stp` | 1 | 1 | 0 | 0 | 6.92 | 6.92 | 0.00 | 6.9 | 깨끗 |
| `out20/print_parts/arm_L_hand.stp` | 8 | 8 | 0 | 0 | 26.93 | 26.31 | 0.61 | 26.9 | 깨끗 |
| `out20/print_parts/arm_L_horn_adapter.stp` | 2 | 2 | 0 | 0 | 2.68 | 2.03 | 0.65 | 2.7 | 예외 |
| `out20/print_parts/arm_L_shell_in.stp` | 7 | 7 | 0 | 0 | 8.79 | 8.64 | 0.15 | 8.8 | 깨끗 |
| `out20/print_parts/arm_L_shell_out.stp` | 3 | 3 | 0 | 0 | 9.90 | 9.73 | 0.16 | 9.9 | 깨끗 |
| `out20/print_parts/arm_R_finger_rod.stp` | 2 | 2 | 0 | 0 | 1.73 | 1.73 | 0.01 | 1.7 | 깨끗 |
| `out20/print_parts/arm_R_flap.stp` | 1 | 1 | 0 | 0 | 6.92 | 6.92 | 0.00 | 6.9 | 깨끗 |
| `out20/print_parts/arm_R_hand.stp` | 8 | 8 | 0 | 0 | 26.93 | 26.31 | 0.61 | 26.9 | 깨끗 |
| `out20/print_parts/arm_R_horn_adapter.stp` | 2 | 2 | 0 | 0 | 2.68 | 2.03 | 0.65 | 2.7 | 예외 |
| `out20/print_parts/arm_R_shell_in.stp` | 7 | 7 | 0 | 0 | 8.79 | 8.64 | 0.15 | 8.8 | 깨끗 |
| `out20/print_parts/arm_R_shell_out.stp` | 3 | 3 | 0 | 0 | 9.90 | 9.73 | 0.16 | 9.9 | 깨끗 |
| `out20/print_parts/camera_clip.stp` | 5 | 5 | 0 | 0 | 0.49 | 0.49 | 0.00 | 0.5 | 깨끗 |
| `out20/print_parts/crest.stp` | 9 | 9 | 0 | 0 | 18.11 | 17.70 | 0.41 | 18.1 | 깨끗 |
| `out20/print_parts/ear_L.stp` | 6 | 6 | 0 | 0 | 19.58 | 19.06 | 0.52 | 19.4 | 깨끗 |
| `out20/print_parts/ear_R.stp` | 6 | 6 | 0 | 0 | 19.58 | 19.06 | 0.52 | 19.4 | 깨끗 |
| `out20/print_parts/ear_disc_L.stp` | 2 | 2 | 0 | 0 | 2.50 | 2.50 | -0.00 | 2.5 | 깨끗 |
| `out20/print_parts/ear_disc_R.stp` | 2 | 2 | 0 | 0 | 2.50 | 2.50 | -0.00 | 2.5 | 깨끗 |
| `out20/print_parts/ear_rocker.stp` | 1 | 1 | 0 | 0 | 0.33 | 0.33 | 0.00 | 0.3 | 깨끗 |
| `out20/print_parts/ear_rod.stp` | 1 | 1 | 0 | 0 | 1.49 | 1.49 | 0.00 | 1.5 | 깨끗 |
| `out20/print_parts/electronics_shelf.stp` | 1 | 1 | 0 | 0 | 30.71 | 30.71 | 0.00 | 30.7 | 깨끗 |
| `out20/print_parts/eye_L.stp` | 2 | 2 | 0 | 0 | 2.07 | 2.07 | 0.00 | 2.1 | 깨끗 |
| `out20/print_parts/eye_R.stp` | 3 | 3 | 0 | 0 | 2.67 | 2.63 | 0.04 | 2.7 | 깨끗 |
| `out20/print_parts/face_plate.stp` | 6 | 6 | 0 | 0 | 31.83 | 31.68 | 0.15 | 31.7 | 깨끗 |
| `out20/print_parts/head_back.stp` | 41 | 41 | 0 | 0 | 189.64 | 184.58 | 5.06 | 189.3 | 깨끗 |
| `out20/print_parts/head_front.stp` | 44 | 44 | 0 | 0 | 96.65 | 94.43 | 2.22 | 96.4 | 깨끗 |
| `out20/print_parts/lid.stp` | 13 | 13 | 0 | 0 | 57.07 | 56.23 | 0.83 | 57.1 | 깨끗 |
| `out20/print_parts/mouth.stp` | 2 | 2 | 0 | 0 | 0.55 | 0.55 | 0.00 | 0.5 | 깨끗 |
| `out20/print_parts/neck_yoke.stp` | 5 | 5 | 0 | 0 | 4.69 | 4.16 | 0.53 | 4.7 | 깨끗 |
| `out20/print_parts/nod_crank.stp` | 2 | 2 | 0 | 0 | 2.02 | 2.02 | -0.00 | 2.0 | 깨끗 |
| `out20/print_parts/nod_rod.stp` | 1 | 1 | 0 | 0 | 0.87 | 0.87 | 0.00 | 0.9 | 깨끗 |
| `out20/print_parts/pelvis_hatch.stp` | 6 | 6 | 0 | 0 | 15.44 | 15.44 | 0.00 | 15.4 | 깨끗 |
| `out20/print_parts/shorts.stp` | 10 | 10 | 0 | 0 | 31.47 | 30.83 | 0.64 | 31.4 | 깨끗 |
| `out20/print_parts/torso_cover_back.stp` | 20 | 20 | 0 | 0 | 55.60 | 54.90 | 0.70 | 55.6 | 깨끗 |
| `out20/print_parts/torso_cover_front.stp` | 20 | 20 | 0 | 0 | 58.93 | 58.16 | 0.77 | 58.9 | 깨끗 |
| `out20/print_parts/torso_frame.stp` | 54 | 54 | 0 | 0 | 225.95 | 224.66 | 1.29 | 225.9 | 깨끗 |
| `out21_arm/print_parts/arm_L_elbow_bracket_in.stp` | 3 | 3 | 0 | 0 | 12.30 | 12.12 | 0.18 | 12.27 | 깨끗 |
| `out21_arm/print_parts/arm_L_elbow_bracket_out.stp` | 2 | 2 | 0 | 0 | 12.31 | 12.29 | 0.02 | 12.28 | 깨끗 |
| `out21_arm/print_parts/arm_L_hand_frame.stp` | 11 | 11 | 0 | 0 | 28.66 | 28.00 | 0.66 | 28.65 | 깨끗 |
| `out21_arm/print_parts/arm_L_horn_adapter.stp` | 2 | 2 | 0 | 0 | 2.66 | 2.64 | 0.01 | 2.63 | 깨끗 |
| `out21_arm/print_parts/arm_L_upper_frame.stp` | 14 | 14 | 0 | 0 | 59.45 | 56.23 | 3.23 | 59.45 | 깨끗 |
| `out21_arm/print_parts/arm_R_elbow_bracket_in.stp` | 3 | 3 | 0 | 0 | 12.30 | 12.12 | 0.18 | 12.27 | 깨끗 |
| `out21_arm/print_parts/arm_R_elbow_bracket_out.stp` | 2 | 2 | 0 | 0 | 12.31 | 12.29 | 0.02 | 12.28 | 깨끗 |
| `out21_arm/print_parts/arm_R_hand_frame.stp` | 11 | 11 | 0 | 0 | 28.66 | 28.00 | 0.66 | 28.65 | 깨끗 |
| `out21_arm/print_parts/arm_R_horn_adapter.stp` | 2 | 2 | 0 | 0 | 2.66 | 2.64 | 0.01 | 2.63 | 깨끗 |
| `out21_arm/print_parts/arm_R_upper_frame.stp` | 14 | 14 | 0 | 0 | 59.45 | 56.23 | 3.23 | 59.45 | 깨끗 |

예외:
- `out20/print_parts/arm_L_horn_adapter.stp` 겹침 (0.65 cm3): v20 팔 옛 판 — 안 뽑음(PRINT_v2 4_팔 은 out21_arm 판). 판(3.5) 위에 원판(2)을 겹쳐 쌓은 설계라 겹침 0.65
- `out20/print_parts/arm_R_horn_adapter.stp` 겹침 (0.65 cm3): v20 팔 옛 판 — 안 뽑음(PRINT_v2 4_팔 은 out21_arm 판). 판(3.5) 위에 원판(2)을 겹쳐 쌓은 설계라 겹침 0.65
- `out20/print_parts/shorts.stp` (판정 아님) 겹침 0.6 은 플랜지 바깥 띠(2 mm 판)가 벽 1.2 mm 안으로 0.6 mm 들어간 것(의도, BODY_HEAD_PRINT_KIT 반바지 절)

부품 64개: 깨끗 62, 예외만 2, 판정 남음 0
