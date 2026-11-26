# v20 오른 다리 STL 확인 (안 A, 허벅지 · 정강이 각 +20 mm, 발판 바깥 10 mm, 혼 구멍 0 · 90°)

- 도구: `print_tests/tools/step2stl.py`(조정자 도구, gmsh/OCC)를 그대로 썼습니다.
- 실행: `LD_LIBRARY_PATH=sim/tmp/glu uv run --no-project --python 3.12 --with gmsh --with numpy python -I print_tests/tools/step2stl.py out20/print_parts/<이름>.stp out20/print_parts_stl/<이름>.stl`
- 기준: 열린 모서리 0, STEP 과 부피 차 1 % 이내. 방향은 `print_parts/parts.json` 의 출력 방향 그대로입니다.

| 부품 | 솔리드 수 | 삼각형 | 열린 모서리 | STL 부피 (cm³) | STEP 부피 (cm³) | 차 | 판정 |
|---|---|---|---|---|---|---|---|
| R_hip_yaw_bracket | 15 | 65932 | 0 | 23.98 | 24.00 | -0.06% | 통과 |
| R_hip_roll_bracket | 11 | 50988 | 0 | 29.89 | 29.89 | +0.00% | 통과 |
| R_thigh_frame | 24 | 98726 | 0 | 30.33 | 30.33 | -0.01% | 통과 |
| R_thigh_shell | 10 | 5040 | 0 | 4.41 | 4.41 | -0.01% | 통과 |
| R_shank_frame | 27 | 102542 | 0 | 29.30 | 29.30 | -0.01% | 통과 |
| R_shank_shell | 9 | 11648 | 0 | 11.76 | 11.76 | -0.02% | 통과 |
| R_ankle_bracket | 15 | 50182 | 0 | 21.53 | 21.54 | -0.06% | 통과 |
| R_foot_sole | 5 | 43816 | 0 | 60.90 | 60.91 | -0.01% | 통과 |

- 솔리드 수가 1보다 큰 것은 한 부품 안의 겹친 덩어리입니다. 슬라이서가 하나로 합칩니다.
