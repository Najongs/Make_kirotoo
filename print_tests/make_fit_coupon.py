"""큐비콘 PLA 공차 시험편 (STEP).

서보 칸 고리: X430 케이스 단면 두 가지(28.5x34, 34x46.5)에 한쪽 여유 c: v2 = 0.10/0.15/0.20, v3 = 0.25/0.30/0.35 mm (v2 결과: 0.20 도 잘 안 들어감, 소유자 2026-11-26).
고리 윗면 바깥 돌기 개수 = 단계(1개 = 가장 작은 여유). COUPON_VER=2|3 (기본 3).
나사 구멍 판: v2 = 2.2/2.4/2.6/2.7/2.9/3.1, v3 = 2.4/2.6/2.8/3.0/3.2/3.4 (왼쪽부터 커짐). 판 전체 약 126x111 mm (작은 베드용). 높이 6 mm, 벽 2.4 mm, 서포트 없음.
나사 구멍 판: M2 통과 2.2/2.4/2.6, M2.5 통과 2.7/2.9/3.1 (왼쪽부터 커짐).
usage: cd sim && uv run python ../print_tests/make_fit_coupon.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code_v17"))
from stepwriter import StepWriter

CASE = {"A_28.5x34": (28.5, 34.0), "B_34x46.5": (34.0, 46.5)}   # X430 케이스 (공식 STEP 치수)
VER = int(os.environ.get("COUPON_VER", "3"))
CLR = {2: [0.10, 0.15, 0.20], 3: [0.25, 0.30, 0.35]}[VER]
GAP = 4.0
H, WALL, R_IN = 6.0, 2.4, 0.5
GREY = (0.6, 0.6, 0.65)

w = StepWriter("fit_coupon_cubicon_pla")
children = []
y0 = 0.0
for name, (a, b) in CASE.items():
    x0 = 0.0
    for k, c in enumerate(CLR):
        ia, ib = a + 2 * c, b + 2 * c
        oa, ob = ia + 2 * WALL, ib + 2 * WALL
        outer = [(0, 0, 2.0), (oa, 0, 2.0), (oa, ob, 2.0), (0, ob, 2.0)]
        inner = [(WALL, WALL, R_IN), (WALL + ia, WALL, R_IN), (WALL + ia, WALL + ib, R_IN), (WALL, WALL + ib, R_IN)]
        sol = [("rprism", ("z", 0.0, H, outer, [inner], []), GREY)]
        for n in range(k + 1):                                   # 단계 표시 돌기
            sol.append(("box", (3.0 + 3.0 * n, 4.5 + 3.0 * n, -1.2, 0.0, H - 2.0, H), GREY))
        part = w.part(f"servo_fit_{name}_c{c:.2f}", sol)
        children.append((part, (x0, y0, 0.0)))
        x0 += oa + GAP
    y0 += b + 2 * WALL + GAP + 1.2

HOLES = {2: [2.2, 2.4, 2.6, 2.7, 2.9, 3.1], 3: [2.4, 2.6, 2.8, 3.0, 3.2, 3.4]}[VER]
holes = [(6 + 9 * i, 6, d / 2) for i, d in enumerate(HOLES)]
plate = w.part("screw_hole_plate_" + "_".join(f"{d:.1f}" for d in HOLES),
               [("rprism", ("z", 0.0, 4.0, [(0, 0, 2), (57, 0, 2), (57, 12, 2), (0, 12, 2)], [], holes), GREY)])
children.append((plate, (0.0, y0, 0.0)))
w.assembly("fit_coupon", children)
out = os.path.join(HERE, f"fit_coupon_v{VER}.stp")
open(out, "w").write(w.text())
print("wrote", out, "entities", w.n)
