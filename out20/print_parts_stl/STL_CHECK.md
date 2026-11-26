# v20 오른 다리 STL 확인

- 도구: `print_tests/tools/step2stl.py`(조정자 도구, gmsh/OCC)를 그대로 썼습니다.
- 실행: `LD_LIBRARY_PATH=sim/tmp/glu uv run --no-project --python 3.12 --with gmsh --with numpy python -I print_tests/tools/step2stl.py out20/print_parts/<이름>.stp out20/print_parts_stl/<이름>.stl`
- 기준: 열린 모서리(open_edges) 0, STEP 과 부피 차 1 % 이내. 방향은 `print_parts/parts.json` 의 출력 방향 그대로입니다.

| 부품 | 솔리드 수 | 삼각형 | 열린 모서리 | STL 부피 (cm³) | STEP 부피 (cm³) | 차 | 판정 |
|---|---|---|---|---|---|---|---|
| R_hip_yaw_bracket | 18 | 97750 | 0 | 47.37 | 47.37 | -0.00% | 통과 |
| R_hip_cover | 6 | 15084 | 0 | 9.00 | 9.00 | -0.00% | 통과 |
| R_hip_roll_plate_back | 9 | 18068 | 0 | 4.16 | 4.16 | +0.01% | 통과 |
| R_hip_roll_plate_front | 5 | 8754 | 0 | 4.17 | 4.17 | +0.00% | 통과 |
| R_thigh_frame | 31 | 91794 | 0 | 23.51 | 23.52 | -0.01% | 통과 |
| R_thigh_shell | 10 | 5040 | 0 | 4.41 | 4.41 | -0.01% | 통과 |
| R_shank_frame | 27 | 97924 | 0 | 22.21 | 22.22 | -0.02% | 통과 |
| R_shank_shell | 10 | 12122 | 0 | 12.45 | 12.45 | -0.01% | 통과 |
| R_ankle_bracket | 15 | 52364 | 0 | 21.49 | 21.51 | -0.06% | 통과 |
| R_foot_sole | 6 | 48324 | 0 | 63.92 | 63.93 | -0.01% | 통과 |

- 솔리드 수가 1보다 큰 것은 한 부품 안의 겹친 덩어리입니다. 슬라이서가 하나로 합칩니다(v19 와 같음).
