import sys, numpy as np
import importlib, os
build5 = importlib.import_module(os.environ.get("BUILD", "build5"))
zt = float(sys.argv[1]); build5.ZT = zt; build5.ZH = zt + 20
import geom4, build4
from geom4 import rot, apply
links, joints, _ = build5.build_model()
J = {j["name"]: j for j in joints}
def solids(l): return [s for pn, ss in links[l]["parts"].items() if "joint_disc" not in pn for s in ss if not (s[0]=="cyl" and s[2]==build4.DISC)]
HP = np.vstack([geom4.samples(s, 0.8) for s in solids("head")])
HP = HP[HP[:, 2] < zt + 5]                       # only the lower half of the head can reach the torso/neck
TS = [(s, build4.aabb(build4.sampled(s))) for s in solids("torso")]
NS = [(s, build4.aabb(build4.sampled(s))) for s in solids("neck")]
def hit(P, M, S):
    Q = apply(M, P); n = 0
    for s, b in S:
        m = np.all((Q >= [x[0] for x in b]) & (Q <= [x[1] for x in b]), axis=1)
        if m.any(): n += int(geom4.inside(s, Q[m], 0.05).sum())
    return n
worst = 0
for pan in range(-90, 91, 15):
    Mp = rot("z", pan, J["head_pan"]["point_mm"])
    for tilt in range(-20, 21, 5):
        worst += hit(HP, Mp @ rot("y", tilt, J["head_tilt"]["point_mm"]), TS)
for tilt in range(-20, 21, 5):
    worst += hit(HP, rot("y", tilt, J["head_tilt"]["point_mm"]), NS)
print(zt, "hits", worst, flush=True)
