"""v22 contact check: zero-pose overlap of every enabled pair in kitoru_v22_rl.xml, and first-contact angles of the
simplified (rl primitives) model against the mesh model (per-part convex hulls) and the shape sweeps.

Run:  cd sim && .venv/bin/python tests/contact_v22.py [--rl model/kitoru_v22_rl.xml] [--mesh model/kitoru_v22.xml]
Out:  stdout + sim/model/contact_v22.json

  * zero: mj_geomDistance of every geom pair MuJoCo's broad phase would test (contype / conaffinity, parent-child
    filter, <exclude>; floor left out) at keyframe zero -> the minimum must be > 0
  * sweeps: one joint (or a joint with others held) stepped 0.5 deg out from 0, first angle where any tested pair has
    distance < 0, refined by bisection to 0.05 deg.
      rl model   : the pairs the rl file enables (what the simulator will feel)
      mesh model : every hull pair between the two body sets regardless of contype bits (the mesh model only lets the
                   legs touch each other); hull pairs already overlapping at the zero pose are left out
  * shape: sim/walk/ranges/results/v22 (joint_ranges.py on build21_arm: point samples inside the part solids, the
    precise reference) single.json and pairs.npz, read if present
"""
import argparse, json, math, os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..")
ap = argparse.ArgumentParser()
ap.add_argument("--rl", default=os.path.join(SIM, "model", "kitoru_v22_rl.xml"))
ap.add_argument("--mesh", default=os.path.join(SIM, "model", "kitoru_v22.xml"))
ap.add_argument("--shape", default=os.path.join(SIM, "walk", "ranges", "results", "v22"))
ap.add_argument("--out", default=os.path.join(SIM, "model", "contact_v22.json"))
A = ap.parse_args()
FT = np.zeros(6)


def enabled_pairs(m):
    """geom pairs MuJoCo's broad phase would test (as check_v21.pairs), floor left out"""
    excl = {(int(m.exclude_signature[i]) >> 16, int(m.exclude_signature[i]) & 0xFFFF) for i in range(m.nexclude)}
    weld, par = m.body_weldid, m.body_parentid
    col = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0]
    out = []
    for i, a in enumerate(col):
        for b in col[i + 1:]:
            ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
            wa, wb = weld[ba], weld[bb]
            if wa == wb or (wa == weld[par[wb]] or wb == weld[par[wa]]):
                continue
            if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
                continue
            if (min(ba, bb), max(ba, bb)) in excl or (max(ba, bb), min(ba, bb)) in excl:
                continue
            out.append((a, b))
    return out


def body_pairs(m, A_, B_, d):
    """every collision geom pair between body sets A_ and B_ (bits ignored), minus pairs overlapping at zero"""
    ga = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in A_ and (m.geom_contype[g] or m.geom_conaffinity[g])]
    gb = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in B_ and (m.geom_contype[g] or m.geom_conaffinity[g])]
    setq(m, d, {})
    return [(a, b) for a in ga for b in gb if mujoco.mj_geomDistance(m, d, a, b, 0.01, FT) > 0]


def setq(m, d, q):
    mujoco.mj_resetDataKeyframe(m, d, m.key("zero").id)
    for k, v in q.items():
        d.qpos[m.joint(k).qposadr[0]] = math.radians(v)
    mujoco.mj_kinematics(m, d)


def gname(m, g):
    return m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}:{m.geom_dataid[g]}"


def min_dist(m, d, prs, q):
    setq(m, d, q)
    best = (1e9, None)
    for a, b in prs:
        v = mujoco.mj_geomDistance(m, d, a, b, 0.05, FT)
        if v < best[0]:
            best = (v, (gname(m, a), gname(m, b)))
    return best


def first_contact(m, d, prs, joint, sign, hold=None, amax=180.0, step=0.5):
    hold = dict(hold or {})
    lo = 0.0
    lim = m.jnt_range[m.joint(joint).id]
    a = step
    while a <= amax + 1e-9:
        q = dict(hold); q[joint] = sign * a
        dd, pr = min_dist(m, d, prs, q)
        if dd < 0:
            hi = a
            for _ in range(4):
                mid = 0.5 * (lo + hi); q[joint] = sign * mid
                if min_dist(m, d, prs, q)[0] < 0:
                    hi = mid
                else:
                    lo = mid
            q[joint] = sign * hi
            return dict(angle=round(sign * hi, 2), pair=min_dist(m, d, prs, q)[1])
        lo = a; a += step
    return dict(angle=None, pair=None)


rl = mujoco.MjModel.from_xml_path(A.rl); rd = mujoco.MjData(rl)
ms = mujoco.MjModel.from_xml_path(A.mesh); md = mujoco.MjData(ms)
out = {"rl": os.path.basename(A.rl), "mesh": os.path.basename(A.mesh)}

# ---------------------------------------------------------------- zero pose
prs = enabled_pairs(rl)
setq(rl, rd, {})
zero = sorted(((mujoco.mj_geomDistance(rl, rd, a, b, 0.2, FT) * 1e3, gname(rl, a), gname(rl, b)) for a, b in prs))
kinds = {}
for a, b in prs:
    k = " - ".join(sorted((rl.body(rl.geom_bodyid[a]).name, rl.body(rl.geom_bodyid[b]).name)))
    kinds[k] = kinds.get(k, 0) + 1
out["zero"] = dict(pairs=len(prs), body_pairs=len(kinds), min_mm=round(zero[0][0], 2), closest=[(round(x, 2), a, b) for x, a, b in zero[:12]],
                   overlapping=[(round(x, 2), a, b) for x, a, b in zero if x <= 0])
print(f"rl enabled non-floor geom pairs {len(prs)} ({len(kinds)} body pairs); zero-pose min distance {zero[0][0]:.2f} mm; "
      f"overlapping {len(out['zero']['overlapping'])}")
for x, a, b in zero[:12]:
    print(f"   {x:7.2f} mm  {a} - {b}")

# ---------------------------------------------------------------- sweeps
LEG = lambda S: [f"{S}_thigh", f"{S}_shank", f"{S}_ankle_link", f"{S}_foot"]
SWEEPS = []
for kn in (0, 30, 60, 90, 110, 117):
    for sg in (-1, 1):
        SWEEPS.append((f"hip_pitch knee {kn}", "R_hip_pitch", sg, {"R_knee": kn}, ["torso"], LEG("R")))
for hp in (-117, -110, -60, 0, 60):
    SWEEPS.append((f"knee hip_pitch {hp}", "R_knee", 1, {"R_hip_pitch": hp}, ["torso", "R_hip_roll_link"], ["R_shank", "R_ankle_link", "R_foot"]))
for rr in (0, 20, -20):
    for sg in (-1, 1):
        SWEEPS.append((f"ankle_pitch roll {rr}", "R_ankle_pitch", sg, {"R_ankle_roll": rr}, ["R_shank"], ["R_foot"]))
for sg in (-1, 1):
    SWEEPS.append(("arm_swing head 0", "R_arm_swing", sg, {}, ["R_arm", "R_forearm"], ["head", "ears", "torso"]))
    SWEEPS.append(("hip_roll arms 0", "R_hip_roll", sg, {}, ["R_hip_roll_link", "R_thigh", "R_shank", "R_ankle_link", "R_foot"],
                   ["R_arm", "R_forearm", "torso", "L_hip_roll_link", "L_thigh", "L_shank", "L_ankle_link", "L_foot"]))
rl_of = {}
res = []
for name, jn, sg, hold, A_, B_ in SWEEPS:
    pr_rl = [(a, b) for a, b in prs if {rl.body(rl.geom_bodyid[a]).name, rl.body(rl.geom_bodyid[b]).name} & set(A_)
             and {rl.body(rl.geom_bodyid[a]).name, rl.body(rl.geom_bodyid[b]).name} & set(B_)]
    pr_ms = body_pairs(ms, A_, B_, md)
    r1 = first_contact(rl, rd, pr_rl, jn, sg, hold) if pr_rl else dict(angle=None, pair="(no enabled pair)")
    r2 = first_contact(ms, md, pr_ms, jn, sg, hold) if pr_ms else dict(angle=None, pair="(no hull pair)")
    row = dict(sweep=name, joint=jn, dir=sg, hold=hold, rl=r1, mesh=r2)
    res.append(row)
    print(f"{name:26s} {'+' if sg > 0 else '-'}  rl {r1['angle']} {r1['pair']}   mesh {r2['angle']} {r2['pair']}")
out["sweeps"] = res

# ---------------------------------------------------------------- shape reference
sp = os.path.join(A.shape, "single.json")
if os.path.exists(sp):
    out["shape_single"] = {f"{r['joint']} {'+' if r['dir'] > 0 else '-'}": dict(limit=r["limit"], first_hit=r["first_hit"], pair=r["pair"])
                           for r in json.load(open(sp)) if r["joint"].startswith("R_")}
pp = os.path.join(A.shape, "pairs.npz")
if os.path.exists(pp):
    P = np.load(pp, allow_pickle=True)
    shp = {}
    for key, a_val, label in (("R 고관절 피치 × 무릎", "knee", "hip_pitch first contact per knee"),):
        Aa, Bb, F = P[f"{key}|A"], P[f"{key}|B"], P[f"{key}|free"]
        rows = {}
        for k, kn in enumerate(Bb):
            if kn < 0:
                continue
            free = Aa[F[:, k]]
            # contiguous free interval around hip pitch 0
            i0 = int(np.argmin(np.abs(Aa)))
            if not F[i0, k]:
                rows[str(kn)] = None; continue
            i = i0
            while i - 1 >= 0 and F[i - 1, k]:
                i -= 1
            j = i0
            while j + 1 < len(Aa) and F[j + 1, k]:
                j += 1
            rows[str(kn)] = (float(Aa[i]), float(Aa[j]), "grid -110 .. 60 step 5")
        shp[label] = rows
    key = "R 발목 피치 × 발목 롤"
    Aa, Bb, F = P[f"{key}|A"], P[f"{key}|B"], P[f"{key}|free"]
    rows = {}
    for k, rr in enumerate(Bb):
        if abs(rr) not in (0, 20):
            continue
        i0 = int(np.argmin(np.abs(Aa)))
        i = i0
        while i - 1 >= 0 and F[i - 1, k]:
            i -= 1
        j = i0
        while j + 1 < len(Aa) and F[j + 1, k]:
            j += 1
        rows[str(rr)] = (float(Aa[i]), float(Aa[j]), "grid step 5")
    shp["ankle_pitch free per ankle roll"] = rows
    out["shape_pairs"] = shp
json.dump(out, open(A.out, "w"), indent=1, ensure_ascii=False, default=float)
print("->", A.out)
