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

# ---- head nod
for tilt in range(-10, 21, 2):
    note("head-torso", (tilt,), hits("head", R("head_nod", tilt), "torso"))
print("head done", len(bad)); sys.stdout.flush()

# ---- arm swing linkage (closed loop): crank angle -> rod and arm poses
import math
from scipy.optimize import brentq
B = build5
C = np.array(B.CRANK); S = np.array([0.0, B.SH_Z])
def arm_angle(Pp, L, ell):
    f = lambda ph: np.hypot(*(S + L * np.array([math.cos(ph), math.sin(ph)]) - Pp)) - ell
    return brentq(f, -0.8, 0.8)
def ang(v): return math.atan2(v[1], v[0])
sides = {"R": (B.R_CRANK, B.R_LEV, 1), "L": (B.L_CRANK, B.L_LEV, -1)}
LINK_POSES = []
for th_deg in np.linspace(-42, 42, 22):
    th = math.radians(th_deg)
    poses_ = {"arm_crank": rot("y", -th_deg, (C[0], 0, C[1]))}
    phis = {}
    for nm, (rc, L, sg) in sides.items():
        P0 = C + sg * np.array([rc, 0.0]); Pp = C + sg * rc * np.array([math.cos(th), math.sin(th)])
        T0 = S + np.array([L, 0.0]); ell = np.hypot(*(T0 - P0))
        ph = arm_angle(Pp, L, ell); phis[nm] = math.degrees(ph)
        Tp = S + L * np.array([math.cos(ph), math.sin(ph)])
        d = math.degrees(ang(Tp - Pp) - ang(T0 - P0))
        Mr = rot("y", -d, (P0[0], 0, P0[1])); Mr[0, 3] += Pp[0] - P0[0]; Mr[2, 3] += Pp[1] - P0[1]
        poses_[f"{nm}_arm_rod"] = Mr
        poses_[f"{nm}_arm"] = rot("y", -math.degrees(ph), (0, 0, B.SH_Z))
    LINK_POSES.append((th_deg, phis))
    mv = list(poses_)
    for a in mv:
        note("arm-torso", (th_deg,), [(f"{a}:{x[0]}", x[1], x[2]) for x in hits(a, poses_[a], "torso")])
    for i, a in enumerate(mv):
        for b in mv[i + 1:]:
            note("arm-arm", (th_deg,), [(f"{a}:{x[0]}", f"{b}:{x[1]}", x[2]) for x in hits(a, poses_[a], b, poses_[b])])
print("arms done", len(bad), "phi range R", round(min(p["R"] for _, p in LINK_POSES), 1), round(max(p["R"] for _, p in LINK_POSES), 1),
      "L", round(min(p["L"] for _, p in LINK_POSES), 1), round(max(p["L"] for _, p in LINK_POSES), 1)); sys.stdout.flush()

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
