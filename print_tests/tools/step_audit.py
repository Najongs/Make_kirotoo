"""부품 STEP 솔리드 점검: 출력 STEP 이 설계대로 읽히는지 본다(gmsh/OpenCASCADE, step2stl.py 와 같은 읽기).

사용: LD_LIBRARY_PATH=sim/tmp/glu uv run -q --no-project --python 3.12 --with gmsh --with numpy \
          python -I print_tests/tools/step_audit.py <STEP 폴더>... [--md 결과.md]
  예: ... step_audit.py out20/print_parts out21_arm/print_parts --md out20/print_parts/STEP_AUDIT.md

부품마다 잰다:
  - brep: 파일 안 MANIFOLD_SOLID_BREP 수(쓴 솔리드 수) · gmsh: 읽힌 3D 솔리드 수 -> 다르면 「솔리드 수」
    (예: 판 밖에 남은 구멍 원이 따로 원기둥 솔리드가 되어 옆 판 구멍을 메움, 2026-11-28 다리 나사 구멍 60개)
  - 음의 부피 솔리드 -> 「음의 부피」(안팎 뒤집힌 솔리드: 외곽선이 스스로 교차하거나 둥글기가 겹침)
  - 겹침 = 솔리드 부피 합 - 합친(union) 부피. 설계 솔리드는 판 위에 보스 · 덧판 · 받침을 겹쳐 쌓고 이음을 0.05 mm
    겹치므로 겹침은 원래 있다(다리 프레임 5-9 %) -> 값은 표에 적고, OVERLAP_REL(15 %) 을 넘을 때만 「겹침」
    (솔리드가 통째로 두 번 들어간 것 같은 큰 중복)
  - 열린 모서리: step2stl.py 와 같은 옵션으로 표면 메시를 만들어 삼각형 둘이 공유하지 않는 모서리 수 -> 0 이 아니면
    「열린 모서리」(외곽선이 스스로 교차하는 판 등, 2026-11-28 뚜껑)
  - parts.json 부피(설계 solid 부피 합, 0.1 cm3 반올림)와 gmsh 솔리드 합의 차이
    -> |차이| > max(VOL_ABS, VOL_REL x 부피) 이면 「부피」
  - union 조각 수(떨어진 덩어리) -> 1 이 아니면 「조각」
EXCEPT 에 적은 부품(경로 끝 「폴더/부품」) · 항목은 이유와 함께 「예외」로 표시하고 판정에서 뺀다.
부품마다 따로 프로세스(시간 제한 TIMEOUT 초)로 잰다. 판정이 하나라도 남으면 종료 코드 1.
"""
import sys, os, json, glob, subprocess, concurrent.futures as cf

OVERLAP_REL = 0.15                          # 겹침 / 합친 부피 (설계 솔리드는 판 위 보스 · 덧판 · 0.05 이음으로 원래 겹친다)
VOL_ABS, VOL_REL = 0.06, 0.01               # cm3 (parts.json 0.1 반올림), 비율
TIMEOUT = 600
# (부품, 항목) -> 이유. 항목: 솔리드 수 · 음의 부피 · 겹침 · 부피 · 조각 · 열린 모서리
# 열쇠: (STEP 경로 끝 「폴더/부품」, 항목) -> 이유
EXCEPT = {
    ("out20/print_parts/arm_L_horn_adapter", "겹침"): "v20 팔 옛 판 — 안 뽑음(PRINT_v2 4_팔 은 out21_arm 판). 판(3.5) 위에 원판(2)을 겹쳐 쌓은 설계라 겹침 0.65",
    ("out20/print_parts/arm_R_horn_adapter", "겹침"): "v20 팔 옛 판 — 안 뽑음(PRINT_v2 4_팔 은 out21_arm 판). 판(3.5) 위에 원판(2)을 겹쳐 쌓은 설계라 겹침 0.65",
}
# 판정은 아니지만 겹침 칸 옆에 이유를 적어 둘 의도된 겹침
NOTE = {"shorts": "겹침 0.6 은 플랜지 바깥 띠(2 mm 판)가 벽 1.2 mm 안으로 0.6 mm 들어간 것(의도, BODY_HEAD_PRINT_KIT 반바지 절)"}

MEASURE = r'''
import sys, json, gmsh
f = sys.argv[1]
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(f); gmsh.model.occ.synchronize()
V = gmsh.model.getEntities(3)
vs = [gmsh.model.occ.getMass(3, t) for _, t in V]
neg = [round(v / 1000, 3) for v in vs if v < 0]
r = dict(gmsh=len(V), sum=sum(vs) / 1000, neg=neg)
if len(V) > 1:
    gmsh.model.occ.fuse([V[0]], V[1:]); gmsh.model.occ.synchronize()
U = gmsh.model.getEntities(3)
r["union"] = sum(gmsh.model.occ.getMass(3, t) for _, t in U) / 1000
r["pieces"] = len(U)
# surface mesh as step2stl.py (same options): edges not shared by exactly 2 triangles
gmsh.clear(); gmsh.model.occ.importShapes(f); gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.model.mesh.generate(2)
types, _, nodes = gmsh.model.mesh.getElements(2)
from collections import Counter
e = Counter()
for t, n in zip(types, nodes):
    if t != 2:
        continue
    for i in range(0, len(n), 3):
        a, b, c = n[i:i + 3]
        for u, v in ((a, b), (b, c), (c, a)):
            e[(min(u, v), max(u, v))] += 1
r["open_edges"] = sum(1 for k in e.values() if k != 2)
gmsh.finalize()
print("RESULT " + json.dumps(r))
'''


def measure(path):
    nb = open(path, errors="replace").read().count("MANIFOLD_SOLID_BREP")
    try:
        p = subprocess.run([sys.executable, "-I", "-c", MEASURE, path], capture_output=True, text=True, timeout=TIMEOUT)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else dict(error=(p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"])
    except subprocess.TimeoutExpired:
        r = dict(error=[f"timeout {TIMEOUT} s"])
    r["brep"] = nb
    return r


def verdict(part, r, ref):
    flags = []
    if "error" in r:
        return [("읽기 실패", " ".join(r["error"]))]
    if r["gmsh"] != r["brep"]:
        flags.append(("솔리드 수", f"brep {r['brep']} -> gmsh {r['gmsh']}"))
    if r["neg"]:
        flags.append(("음의 부피", f"{len(r['neg'])}개 {r['neg'][:3]}"))
    ov = r["sum"] - r["union"]
    if ov > OVERLAP_REL * abs(r["union"]):
        flags.append(("겹침", f"{ov:.2f} cm3"))
    if r["open_edges"]:
        flags.append(("열린 모서리", str(r["open_edges"])))
    if ref is not None and abs(r["sum"] - ref) > max(VOL_ABS, VOL_REL * abs(ref)):
        flags.append(("부피", f"parts.json {ref} vs 솔리드 합 {r['sum']:.2f}"))
    if r["pieces"] != 1:
        flags.append(("조각", f"합친 뒤 {r['pieces']}덩어리"))
    return flags


def main():
    args = sys.argv[1:]
    md = None
    if "--md" in args:
        i = args.index("--md"); md = args[i + 1]; del args[i:i + 2]
    files, refs = [], {}
    for d in args:
        pj = os.path.join(d, "parts.json")
        ref = {x["part"]: x["volume_cm3"] for x in json.load(open(pj))} if os.path.exists(pj) else {}
        for f in sorted(glob.glob(os.path.join(d, "*.stp"))):
            files.append(f); refs[f] = ref.get(os.path.basename(f)[:-4])
    with cf.ThreadPoolExecutor(int(os.environ.get("JOBS", "16"))) as ex:
        res = dict(zip(files, ex.map(measure, files)))
    rows, bad, exc = [], 0, 0
    for f in files:
        part = os.path.basename(f)[:-4]; r = res[f]
        flags = verdict(part, r, refs[f])
        why = {k: w for (tail, k), w in EXCEPT.items() if os.path.abspath(f)[:-4].endswith(os.sep + tail)}
        hard = [(k, v) for k, v in flags if k not in why]
        soft = [(k, v, why[k]) for k, v in flags if k in why]
        bad += bool(hard); exc += bool(soft)
        rows.append((os.path.relpath(f), r, refs[f], hard, soft))
    out = ["| 부품 | brep | gmsh | 음의 부피 | 열린 모서리 | 솔리드 합 cm3 | 합친 부피 cm3 | 겹침 cm3 | parts.json cm3 | 판정 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for f, r, ref, hard, soft in rows:
        if "error" in r:
            out.append(f"| `{f}` | {r['brep']} | - | - | - | - | - | - | {ref} | **읽기 실패** {' '.join(r['error'])} |"); continue
        j = "; ".join(f"**{k}** {v}" for k, v in hard) or ("예외" if soft else "깨끗")
        out.append(f"| `{f}` | {r['brep']} | {r['gmsh']} | {len(r['neg'])} | {r['open_edges']} | {r['sum']:.2f} | {r['union']:.2f} | "
                   f"{r['sum'] - r['union']:.2f} | {ref if ref is not None else '-'} | {j} |")
    ex_lines = [f"- `{f}` {k} ({v}): {why}" for f, r, ref, hard, soft in rows for k, v, why in soft]
    ex_lines += [f"- `{f}` (판정 아님) {NOTE[os.path.basename(f)[:-4]]}" for f, *_ in rows if os.path.basename(f)[:-4] in NOTE]
    summary = f"부품 {len(rows)}개: 깨끗 {len(rows) - bad - sum(1 for *_, h, s in rows if s and not h)}, 예외만 {sum(1 for *_, h, s in rows if s and not h)}, 판정 남음 {bad}"
    text = "\n".join(out + ["", "예외:"] + (ex_lines or ["- 없음"]) + ["", summary])
    print(text)
    if md:
        head = ["# STEP 솔리드 점검", "",
                "`print_tests/tools/step_audit.py` 가 쓴 표(gmsh/OpenCASCADE 로 STEP 을 읽어 잼). 기준: brep = gmsh 솔리드 수, "
                "음의 부피 0, 표면 메시 열린 모서리 0(step2stl.py 와 같은 옵션), 겹침(솔리드 합 − 합친 부피) ≤ 15 %(설계 솔리드는 보스 · 덧판 · 0.05 이음으로 원래 겹침), parts.json 부피와 솔리드 합 차 ≤ max(0.06 cm³, 1 %), "
                "합친 뒤 한 덩어리.", "",
                "다시 재기: `LD_LIBRARY_PATH=sim/tmp/glu uv run -q --no-project --python 3.12 --with gmsh --with numpy python -I "
                "print_tests/tools/step_audit.py " + " ".join(os.path.relpath(a) for a in args) + f" --md {os.path.relpath(md)}`", ""]
        open(md, "w").write("\n".join(head) + text + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
