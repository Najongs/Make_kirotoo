"""Pose sweeps for v4: head pan x tilt against neck/torso, and hip yaw x hip pitch (x knee) against the torso."""
import itertools, numpy as np
import geom4, build4
from geom4 import rot, apply

links, joints, _ = build4.build_model()
J = {j["name"]: j for j in joints}
STEP = 0.8


def solids(link, skip=("joint_disc",)):
    return [(pn, s) for pn, ss in links[link]["parts"].items() if not any(k in pn for k in skip) for s in ss]


def pts(link):
    return [(pn, geom4.samples(s, STEP)) for pn, s in solids(link)]


FIX = {ln: solids(ln) for ln in links}
BB = {id(s): build4.aabb(build4.sampled(s)) for ss in FIX.values() for _, s in ss}


def hits(moving, M, fixed_link, margin=0.05):
    out = []
    for pn, P in moving:
        Q = apply(M, P)
        qlo, qhi = Q.min(0), Q.max(0)
        for fn, f in FIX[fixed_link]:
            b = BB[id(f)]
            if any(qhi[i] < b[i][0] or qlo[i] > b[i][1] for i in range(3)):
                continue
            m = np.all((Q >= [x[0] for x in b]) & (Q <= [x[1] for x in b]), axis=1)
            if not m.any():
                continue
            n = int(geom4.inside(f, Q[m], margin).sum())
            if n:
                out.append((pn, fn, n))
    return out


report = {}
# ---- head ----
head, neck = pts("head"), pts("neck")
zt = J["head_tilt"]["point_mm"]; zp = J["head_pan"]["point_mm"]
bad = []
for pan in range(-90, 91, 10):
    Mp = rot("z", pan, zp)
    bad += [("neck", pan, None, h) for h in hits(neck, Mp, "torso")]
    for tilt in range(-20, 21, 4):
        Mt = Mp @ rot("y", tilt, zt)
        bad += [("head", pan, tilt, h) for h in hits(head, Mt, "torso")]
        if pan == 0:
            bad += [("head-neck", 0, tilt, h) for h in hits(head, rot("y", tilt, zt), "neck")]
print("head pan x tilt sweep hits:", len(bad))
for b in bad[:12]: print("  ", b)

# ---- legs (right side; left is the mirror) ----
yaw_l, thigh, shank = pts("R_hip_yaw_link"), pts("R_thigh"), pts("R_shank")
py, pp, pk = J["R_hip_yaw"]["point_mm"], J["R_hip_pitch"]["point_mm"], J["R_knee"]["point_mm"]
lbad = []
for yaw in range(-30, 31, 10):
    My = rot("z", yaw, py)
    lbad += [("yaw_link", yaw, None, None, h) for h in hits(yaw_l, My, "torso")]
    for pitch in range(-90, 31, 15):
        Mpch = My @ rot("y", pitch, pp)
        lbad += [("thigh", yaw, pitch, None, h) for h in hits(thigh, Mpch, "torso")]
        for knee in (0, 60, 130):
            Mk = Mpch @ rot("y", knee, pk)
            lbad += [("shank", yaw, pitch, knee, h) for h in hits(shank, Mk, "torso")]
print("leg yaw x pitch x knee sweep hits vs torso:", len(lbad))
seen = set()
for b in lbad:
    k = (b[0], b[4][0], b[4][1])
    if k not in seen:
        seen.add(k); print("  ", b)
