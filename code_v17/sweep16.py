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

# ---- head nod (closed loop: head servo crank + rod to the yoke anchor)
import math
from scipy.optimize import brentq
B = build5
P_ = np.array([0.0, B.ZT9]); S0 = np.array(B.NOD_S); A_ = np.array(B.NOD_A)
pin0 = S0 + np.array([B.NOD_R, 0.0]); ell_n = np.hypot(*(pin0 - A_))
def rot2(v, deg, c):           # rot() convention in the x-z plane: +deg turns +x toward -z
    a = math.radians(deg); d = v - c
    return c + np.array([d[0] * math.cos(a) + d[1] * math.sin(a), -d[0] * math.sin(a) + d[1] * math.cos(a)])
NOD = []
import build16 as B16
E0 = np.array(B16.EAR_C); P1, P2, LE = B16.ear_geometry(); P1 = np.array(P1); P2 = np.array(P2)
def angd(u): return math.degrees(math.atan2(u[1], u[0]))
HEADPOSE = {}
EARS = []
for beta in sorted(set(list(np.linspace(-10, 20, 16)) + [-10.0, 20.0])):
    Sw = rot2(S0, beta, P_)
    f = lambda al: np.hypot(*(rot2(rot2(pin0, al, S0), beta, P_) - A_)) - ell_n
    al = brentq(f, -45, 70)
    pw = rot2(rot2(pin0, al, S0), beta, P_)
    d = -math.degrees(math.atan2(*(pw - A_)[::-1]) - math.atan2(*(pin0 - A_)[::-1]))
    Mh = R("head_nod", beta)
    Mc = Mh @ rot("y", al, (S0[0], 0, S0[1]))
    Mr = rot("y", d, (A_[0], 0, A_[1]))
    NOD.append((round(beta, 1), round(al, 1)))
    note("head-torso", (beta,), hits("head", Mh, "torso"))
    note("crank-torso", (beta,), hits("nod_crank", Mc, "torso"))
    note("rod-head", (beta,), hits("nod_rod", Mr, "head", Mh))
    note("rod-crank", (beta,), [x for x in hits("nod_rod", Mr, "nod_crank", Mc) if "pin" not in x[1]])
    note("rod-torso", (beta,), [x for x in hits("nod_rod", Mr, "torso") if "anchor_pin" not in x[1]])
    q1 = rot2(P1, al, S0)
    ps = brentq(lambda p: np.hypot(*(rot2(P2, p, E0) - q1)) - LE, -89, 89)
    q2 = rot2(P2, ps, E0)
    ph = -(angd(q2 - q1) - angd(P2 - P1))
    Me = Mh @ rot("y", ps, (E0[0], 0, E0[1]))
    Mer = rot("y", ph, (P1[0], 0, P1[1])); Mer[0, 3] += q1[0] - P1[0]; Mer[2, 3] += q1[1] - P1[1]; Mer = Mh @ Mer
    HEADPOSE[round(float(beta))] = (Mh, Me)
    EARS.append((round(float(beta), 1), round(ps, 1)))
    note("ears-head", (beta,), [x for x in hits("ears", Me, "head", Mh) if not ("axle" in x[0] and "bearing" in x[1])])
    note("ears-torso", (beta,), hits("ears", Me, "torso"))
    note("ears-crank", (beta,), hits("ears", Me, "nod_crank", Mc))
    note("erod-head", (beta,), hits("ear_rod", Mer, "head", Mh))
    note("erod-crank", (beta,), [x for x in hits("ear_rod", Mer, "nod_crank", Mc) if "pin" not in x[1]])
    note("erod-ears", (beta,), [x for x in hits("ear_rod", Mer, "ears", Me) if "pin" not in x[1]])
    note("erod-nodrod", (beta,), hits("ear_rod", Mer, "nod_rod", Mr))
    note("erod-torso", (beta,), hits("ear_rod", Mer, "torso"))
print("head done", len(bad), "crank angle for head -10..20:", NOD[0][1], "..", NOD[-1][1], "ears", EARS[0][1], "..", EARS[-1][1]); sys.stdout.flush()

# ---- arms: one servo per arm, free rotation about the shoulder axis -> find the collision-free range
import build16 as B16
def arm_hits(nm, a, Mh=None, Me=None):
    M = R(f"{nm}_arm", a) if False else R(f"{nm}_arm_swing", a)
    h = hits(f"{nm}_arm", M, "torso")
    for ln in [l for l in links if l.startswith(nm + "_") and l != f"{nm}_arm"]:
        h += hits(f"{nm}_arm", M, ln)
    if Mh is not None:
        h += hits(f"{nm}_arm", M, "head", Mh)
        h += hits(f"{nm}_arm", M, "ears", Me)
    return h
ARM_RANGE = {}
for nm in ("R", "L"):
    ok = {}
    for a in range(-180, 181, 5):
        hh = arm_hits(nm, a)
        for beta in (-10, 20):
            Mh, Me = HEADPOSE[beta]
            hh += arm_hits(nm, a, Mh, Me)
        ok[a] = not hh
    lo = 0
    while lo - 5 >= -180 and ok[lo - 5]: lo -= 5
    hi = 0
    while hi + 5 <= 180 and ok[hi + 5]: hi += 5
    ARM_RANGE[nm] = (lo, hi)
    for a in range(lo, hi + 1, 5):
        note("arm-check", (a,), arm_hits(nm, a))
print("arms done", len(bad), "free range", ARM_RANGE); sys.stdout.flush()

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
