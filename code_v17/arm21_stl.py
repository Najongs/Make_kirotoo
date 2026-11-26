"""out21_arm/print_parts/*.stp -> out21_arm/print_parts_stl/*.stl + STL_CHECK.md (print_tests/tools/step2stl.py, gmsh).

usage (repo root):  python3 code_v17/arm21_stl.py
Needs uv + the libGLU copy in sim/tmp/glu (see the head of step2stl.py). Pass: open edges 0, |STL - STEP volume| <= 1 %.
"""
import os, re, json, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "out21_arm", "print_parts")
DST = os.path.join(ROOT, "out21_arm", "print_parts_stl")
os.makedirs(DST, exist_ok=True)
parts = json.load(open(os.path.join(SRC, "parts.json")))
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(ROOT, "sim", "tmp", "glu"))
rows, bad = [], 0
for p in sorted(parts, key=lambda p: (p["part"][4] != "R", p["part"])):
    n = p["part"]
    cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", "gmsh", "--with", "numpy", "python", "-I",
           os.path.join(ROOT, "print_tests", "tools", "step2stl.py"), os.path.join(SRC, n + ".stp"), os.path.join(DST, n + ".stl")]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()[-1]
    m = re.search(r"solids (\d+) tris (\d+) open_edges (\d+) vol_stl ([\d.]+) cm3 vol_step ([\d.]+) cm3 diff ([+-][\d.]+)%", out)
    sol, tri, oe, vs, vp, d = m.groups()
    ok = int(oe) == 0 and abs(float(d)) <= 1.0
    bad += not ok
    rows.append(f"| {n} | {p['up']} | {sol} | {tri} | {oe} | {vs} | {vp} | {d}% | {'통과' if ok else '실패'} |")
    print(out); sys.stdout.flush()
md = ["# v21 팔꿈치 팔 STL 확인 (제안, 채택 전)", "",
      "- 도구: `print_tests/tools/step2stl.py`(gmsh/OpenCASCADE) 그대로. 실행: `python3 code_v17/arm21_stl.py`(저장소 맨 위에서).",
      "- 기준: 열린 모서리 0, STEP 과 부피 차 1 % 이내. 방향은 `print_parts/parts.json` 의 출력 방향(「위」)이 이미 들어가 있습니다.",
      "- 솔리드 수가 1보다 큰 것은 한 부품 안에서 겹친 덩어리입니다. 슬라이서가 하나로 합칩니다(v20 다리 STL 과 같음).", "",
      "| 부품 | 위 | 솔리드 수 | 삼각형 | 열린 모서리 | STL 부피 (cm³) | STEP 부피 (cm³) | 차 | 판정 |",
      "|---|---|---|---|---|---|---|---|---|"] + rows + ["", f"실패 {bad} 개."]
open(os.path.join(DST, "STL_CHECK.md"), "w").write("\n".join(md) + "\n")
print("failed", bad)
sys.exit(1 if bad else 0)
