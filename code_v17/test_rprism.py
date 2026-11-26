import re, math, subprocess
from stepwriter import StepWriter
from shapes import rrect, annulus_sector, circle_loop, sample, segments
w = StepWriter("t.stp")
w.part("t", [
  ("rprism", ("z", 0, 5, rrect(-60, 60, -35, 35, 18), [], [(-40, -7, 1.7), (-40, 7, 1.7)]), (1, .5, 0)),
  ("rprism", ("y", -22, 22, annulus_sector(392, 0, 41.5, 44, 10, 190), [], []), (1, .5, 0)),   # (u,v)=(z,x)
  ("rprism", ("z", 374, 414, rrect(-28, 34, -34, 34, 24), [rrect(-26, 32, -32, 32, 22)], []), (1, .5, 0)),
  ("rprism", ("z", 226, 233, circle_loop(0, -40, 18.5), [], [(0, -40, 12.5)]), (.8, .8, .8)),
])
open("out3/test_r.stp", "w").write(w.text())
# geometric check: arc-edge vertices must lie on their circle
t = open("out3/test_r.stp").read().split("DATA;")[1].split("ENDSEC;")[0]
D = {int(m.group(1)): m.group(2) for m in re.finditer(r"#(\d+)=(.*?);\n", t + "\n")}
refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", re.sub(r"'[^']*'", "", s))]
pt = lambda i: tuple(float(x) for x in re.search(r"\(([^()]*)\)\)$", D[i]).group(1).split(","))
worst = 0; narcs = 0
for i, v in D.items():
    if v.startswith("EDGE_CURVE"):
        va, vb, cv = refs(v)[:3]
        if D[cv].startswith("CIRCLE"):
            ax = refs(D[cv])[0]; c = pt(refs(D[ax])[0]); r = float(D[cv].rsplit(",", 1)[1].rstrip(")"))
            for vv in (va, vb):
                p = pt(refs(D[vv])[0]); worst = max(worst, abs(math.dist(p, c) - r)); narcs += 1
print("circle-vertex checks", narcs, "worst radial error mm", round(worst, 9))
print(subprocess.run(["python3", "check_step.py", "out3/test_r.stp"], capture_output=True, text=True).stdout)
print("C-sector segments:", [s[0] for s in segments(annulus_sector(392, 0, 41.5, 44, 10, 190))])
