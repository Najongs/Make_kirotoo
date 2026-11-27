"""First-contact angles on the design shape (build21_arm part solids) for the sweeps of contact_v22.py: hip pitch at
knee 0, knee at fixed hip pitch, ankle pitch at ankle roll 0 / +-20, hip roll outward with the arms at 0.

Method (as sim/walk/ranges/joint_ranges.py): grid points inside the solids of one link (geom4.samples, STEP mm),
tested inside the solids of the other link (geom4.inside, margin 0.05 mm), both ways round so thin walls are not
missed; 0.5 deg steps out from 0, then bisection to about 0.03 deg. Pairs overlapping at the zero pose are left out.

Run:  cd code_v17 && ../sim/.venv/bin/python ../sim/tests/shape_contact_v22.py      (model build 1.5 min + sweeps)
Out:  stdout + sim/model/shape_contact_v22.json
"""
import os, sys, json
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code_v17")
sys.path.insert(0, CODE)
os.chdir(CODE)
import geom4, build4
from geom4 import rot, apply
import build21_arm as BM

STEP = 1.0
LINKS, JOINTS, _ = BM.build_model()
J = {j["name"]: j for j in JOINTS}
CHILD = {j["child"]: j for j in JOINTS}


def path(ln):
    p = []
    while ln in CHILD:
        j = CHILD[ln]; p.append(j["name"]); ln = j["parent"]
    return p[::-1]


def pose(ln, q):
    M = np.eye(4)
    for jn in path(ln):
        M = M @ rot(J[jn]["axis"], q.get(jn, 0.0), J[jn]["point_mm"])
    return M


def solids(ln):
    return [(pn, s) for pn, ss in LINKS[ln]["parts"].items() if "joint_disc" not in pn for s in ss
            if not (s[0] == "cyl" and s[2] == build4.DISC)]


SOL = {ln: solids(ln) for ln in LINKS}
BB = {id(s): build4.aabb(build4.sampled(s)) for ss in SOL.values() for _, s in ss}
PTS = {ln: [(pn, geom4.samples(s, STEP)) for pn, s in SOL[ln]] for ln in LINKS}


def hits(a, Ma, b, Mb, base=()):
    out = []
    Mi = np.linalg.inv(Mb) @ Ma
    for pn, P in PTS[a]:
        if not len(P):
            continue
        Q = apply(Mi, P); qlo, qhi = Q.min(0), Q.max(0)
        for fn, f in SOL[b]:
            bb = BB[id(f)]
            if any(qhi[i] < bb[i][0] or qlo[i] > bb[i][1] for i in range(3)) or (a, pn, b, fn) in base:
                continue
            sel = np.all((Q >= [x[0] for x in bb]) & (Q <= [x[1] for x in bb]), axis=1)
            if sel.any():
                n = int(geom4.inside(f, Q[sel], 0.05).sum())
                if n:
                    out.append((a, pn, b, fn, n))
    return out


def collide(q, A, B, base=()):
    res = []
    for a in A:
        for b in B:
            Ma, Mb = pose(a, q), pose(b, q)
            res += hits(a, Ma, b, Mb, base) + hits(b, Mb, a, Ma, base)
    return res


def baseline(A, B):
    return {(x[0], x[1], x[2], x[3]) for x in collide({}, A, B)}


def sweep(args):
    name, jn, sg, hold, A, B = args
    base = baseline(A, B)
    lo, a = 0.0, 0.5
    while a <= 180:
        q = dict(hold); q[jn] = sg * a
        if collide(q, A, B, base):
            hi = a
            for _ in range(4):
                mid = 0.5 * (lo + hi); q[jn] = sg * mid
                if collide(q, A, B, base):
                    hi = mid
                else:
                    lo = mid
            q[jn] = sg * hi
            h = max(collide(q, A, B, base), key=lambda x: x[4])
            return dict(sweep=name, joint=jn, dir=sg, hold=hold, free=round(sg * lo, 2), first_hit=round(sg * hi, 2),
                        pair=f"{h[0]}:{h[1]} <-> {h[2]}:{h[3]}", baseline=len(base))
        lo = a; a += 0.5
    return dict(sweep=name, joint=jn, dir=sg, hold=hold, free=None, first_hit=None, pair=None, baseline=len(base))


LEG = ["R_thigh", "R_shank", "R_ankle_link", "R_foot"]
TASKS = []
for sg in (-1, 1):
    TASKS.append(("hip_pitch knee 0", "R_hip_pitch", sg, {}, ["torso"], LEG))
for hp in (-117, -110, -60, 0, 60):
    TASKS.append((f"knee hip_pitch {hp}", "R_knee", 1, {"R_hip_pitch": hp}, ["torso", "R_hip_roll_link", "R_thigh"],
                  ["R_shank", "R_ankle_link", "R_foot"]))
for rr in (0, 20, -20):
    for sg in (-1, 1):
        TASKS.append((f"ankle_pitch roll {rr}", "R_ankle_pitch", sg, {"R_ankle_roll": rr}, ["R_shank"], ["R_foot"]))
TASKS.append(("hip_roll out (arms 0)", "R_hip_roll", -1, {}, ["R_arm", "R_forearm", "torso"], LEG + ["R_hip_roll_link"]))

if __name__ == "__main__":
    with Pool(len(TASKS)) as p:
        res = p.map(sweep, TASKS)
    for r in res:
        print(f"{r['sweep']:24s} {'+' if r['dir'] > 0 else '-'}  first hit {r['first_hit']}  {r['pair']}")
    json.dump(dict(step_mm=STEP, build="build21_arm", sweeps=res),
              open(os.path.join(HERE, "..", "model", "shape_contact_v22.json"), "w"), indent=1)
