"""Pose sweeps for v5: each joint's moving subtree against the links on its parent side.
Head: pan x tilt vs torso and neck. Leg: yaw, pitch, knee, ankle pitch, roll in coupled grids."""
import itertools, sys, numpy as np
import geom4, build4, importlib, os
build5 = importlib.import_module(os.environ.get("BUILD", "build5"))
from geom4 import rot, apply

links, joints, _ = build5.build_model()
J = {j["name"]: j for j in joints}
STEP = 0.6


def solids(link):
    return [(pn, s) for pn, ss in links[link]["parts"].items() if "joint_disc" not in pn for s in ss
            if not (s[0] == "cyl" and s[2] == build4.DISC)]


PTS = {}
def pts(link):
    if link not in PTS:
        PTS[link] = [(pn, geom4.samples(s, STEP)) for pn, s in solids(link)]
    return PTS[link]


FIX = {ln: solids(ln) for ln in links}
BB = {id(s): build4.aabb(build4.sampled(s)) for ss in FIX.values() for _, s in ss}


def hits(moving_link, M, fixed_link, Mf=None, margin=0.05):
    """Points of moving_link (pose M) inside fixed_link (pose Mf, default identity)."""
    out = []
    Mi = np.linalg.inv(Mf) @ M if Mf is not None else M
    for pn, P in pts(moving_link):
        Q = apply(Mi, P)
        qlo, qhi = Q.min(0), Q.max(0)
        for fn, f in FIX[fixed_link]:
            b = BB[id(f)]
            if any(qhi[i] < b[i][0] or qlo[i] > b[i][1] for i in range(3)):
                continue
            m = np.all((Q >= [x[0] for x in b]) & (Q <= [x[1] for x in b]), axis=1)
            if m.any():
                n = int(geom4.inside(f, Q[m], margin).sum())
                if n:
                    out.append((pn, fn, n))
    return out


def R(name, deg):
    j = J[name]
    return rot(j["axis"], deg, j["point_mm"])


bad = []
def note(tag, pose, h):
    for x in h:
        bad.append((tag, pose, x))

# ---- head
for pan in range(-90, 91, 10):
    Mp = R("head_pan", pan)
    note("neck-torso", (pan,), hits("neck", Mp, "torso"))
    for tilt in range(-20, 21, 4):
        Mt = Mp @ R("head_tilt", tilt)
        note("head-torso", (pan, tilt), hits("head", Mt, "torso"))
        if pan == 0:
            note("head-neck", (tilt,), hits("head", R("head_tilt", tilt), "neck"))
print("head done", len(bad)); sys.stdout.flush()

# ---- right leg chain (left is its mirror)
S = "R"
names = [f"{S}_hip_yaw", f"{S}_hip_pitch", f"{S}_knee", f"{S}_ankle_pitch", f"{S}_ankle_roll"]
chain = ["torso", f"{S}_hip_yaw_link", f"{S}_thigh", f"{S}_shank", f"{S}_ankle_link", f"{S}_foot"]
rng = {n: (J[n]["lower_deg"], J[n]["upper_deg"]) for n in names}


def grid(n, k):
    lo, hi = rng[n]
    return list(np.linspace(lo, hi, k))


def poses(link_idx, angles):
    """World pose of chain[link_idx] for joint angles (list for joints 0..link_idx-1)."""
    M = np.eye(4)
    for n, a in zip(names[:link_idx], angles):
        M = M @ R(n, a)
    return M

# yaw link vs torso
for y in grid(names[0], 13):
    note("yaw-torso", (y,), hits(chain[1], poses(1, [y]), "torso"))
# thigh vs torso, yaw link
for y, p in itertools.product(grid(names[0], 5), grid(names[1], 13)):
    M = poses(2, [y, p])
    note("thigh-torso", (y, p), hits(chain[2], M, "torso"))
    note("thigh-yaw", (y, p), hits(chain[2], M, chain[1], poses(1, [y])))
print("thigh done", len(bad)); sys.stdout.flush()
# shank vs thigh (knee only), shank vs torso/yaw link (coarse)
for k in grid(names[2], 27):
    note("shank-thigh", (k,), hits(chain[3], R(names[2], k), chain[2]))
for y, p, k in itertools.product(grid(names[0], 3), grid(names[1], 5), grid(names[2], 5)):
    M = poses(3, [y, p, k])
    note("shank-torso", (y, p, k), hits(chain[3], M, "torso"))
    note("shank-yaw", (y, p, k), hits(chain[3], M, chain[1], poses(1, [y])))
print("shank done", len(bad)); sys.stdout.flush()
# ankle link vs shank, thigh
for k, a in itertools.product(grid(names[2], 6), grid(names[3], 22)):
    Mk = R(names[2], k)
    M = Mk @ R(names[3], a)
    note("ankle-shank", (k, a), hits(chain[4], M, chain[3], Mk))
    note("ankle-thigh", (k, a), hits(chain[4], M, chain[2]))
# foot vs ankle link, shank, thigh
for k, a, r in itertools.product(grid(names[2], 4), grid(names[3], 12), grid(names[4], 13)):
    Mk = R(names[2], k)
    Ma = Mk @ R(names[3], a)
    M = Ma @ R(names[4], r)
    if k == rng[names[2]][0]:
        note("foot-ankle", (a, r), hits(chain[5], M, chain[4], Ma))
    note("foot-shank", (k, a, r), hits(chain[5], M, chain[3], Mk))
    note("foot-thigh", (k, a, r), hits(chain[5], M, chain[2]))
print("leg done", len(bad))

seen = {}
for tag, pose, (pn, fn, n) in bad:
    key = (tag, pn, fn)
    seen.setdefault(key, []).append((tuple(round(float(x)) for x in pose), n))
for key, v in seen.items():
    worst = max(v, key=lambda t: t[1])
    print(key, "poses:", len(v), "first:", v[0][0], "last:", v[-1][0], "worst:", worst)
