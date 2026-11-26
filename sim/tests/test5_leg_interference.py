"""Test 5: interference between the two legs over a joint grid (torso fixed, kinematics only).

Grid: hip yaw R x hip yaw L in -30..30 (7 x 7, same / opposite directions) x hip pitch scissor (R = p, L = -p,
p in -30..30) x knee (both) with the ankle pitch keeping the feet flat (ankle = -(hip + knee)).
Coarse: every collision hull of the R leg against every hull of the L leg (thigh, hip yaw link included;
in the model only shank/ankle/foot carry the leg-leg bit). Fine: configurations where hulls touch are re-checked
on the design solids (geom4.inside on sampled points, render17 link poses), because the hulls are larger than the parts.

Run: cd sim && uv run python tests/test5_leg_interference.py -> results/test5_leg_interference.json, .png
"""
import itertools, math, numpy as np
from common import *
import render17 as R17
import geom4, build4
r6 = R17.r

m, d = load(fixed=True)
LEGL = ["hip_yaw_link", "thigh", "shank", "ankle_link", "foot"]
for i in range(m.ngeom):
    b = body_of_geom(m, i)
    if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
        continue
    if b and b[:2] in ("R_", "L_") and b[2:] in LEGL:
        m.geom_contype[i], m.geom_conaffinity[i] = (4, 8) if b[0] == "R" else (8, 4)
    else:
        m.geom_contype[i] = m.geom_conaffinity[i] = 0

YAW = [-30, -20, -10, 0, 10, 20, 30]; PITCH = [-30, -15, 0, 15, 30]; KNEE = [0, 30, 60]


def cat(b1, b2):
    s = sorted([b1[2:], b2[2:]])
    if s == ["foot", "foot"]: return "foot-foot"
    if s == ["shank", "shank"]: return "shank-shank"
    if s == ["foot", "shank"]: return "foot-shank"
    return "-".join(s)


def angles(yr, yl, p, k):
    a = {"R_hip_yaw": yr, "L_hip_yaw": yl, "R_hip_pitch": p, "L_hip_pitch": -p, "R_knee": k, "L_knee": k,
         "R_ankle_pitch": -(p + k), "L_ankle_pitch": -(-p + k), "R_ankle_roll": 0, "L_ankle_roll": 0}
    return a


# design point samples per leg link (no joint discs)
STEP = 1.5
def solids(ln):
    return [(pn, s) for pn, ss in r6.links[ln]["parts"].items() if "joint_disc" not in pn for s in ss
            if not (s[0] == "cyl" and tuple(s[2]) == tuple(build4.DISC))]
PTS, SOL = {}, {}
for S_ in "RL":
    for l in LEGL:
        ln = f"{S_}_{l}"
        SOL[ln] = [(pn, s, build4.aabb(build4.sampled(s))) for pn, s in solids(ln)]
        PTS[ln] = np.concatenate([geom4.samples(s, STEP) for _, s in solids(ln)])


def precise(ln_a, ln_b, ang, margin=0.1):
    """number of sampled points of ln_a inside ln_b's solids at design pose `ang` (deg)"""
    Ma, Mb = r6.link_pose(ln_a, ang), r6.link_pose(ln_b, ang)
    Mi = np.linalg.inv(Mb) @ Ma
    Q = PTS[ln_a] @ Mi[:3, :3].T + Mi[:3, 3]
    n = 0; parts = set()
    for pn, s, bb in SOL[ln_b]:
        lo = np.array([x[0] for x in bb]); hi = np.array([x[1] for x in bb])
        sel = np.all((Q >= lo) & (Q <= hi), axis=1)
        if sel.any():
            k = int(geom4.inside(s, Q[sel], margin).sum())
            if k:
                n += k; parts.add(pn)
    return n, sorted(parts)


GRID = {}; rows = []
for p, k in itertools.product(PITCH, KNEE):
    for yr, yl in itertools.product(YAW, YAW):
        ang = angles(yr, yl, p, k)
        q = m.qpos0.copy(); set_joints(m, q, {n: math.radians(v) for n, v in ang.items()})
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        pairs = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = body_of_geom(m, c.geom1), body_of_geom(m, c.geom2)
            key = tuple(sorted([b1, b2]))
            pairs[key] = min(pairs.get(key, 1.0), c.dist)
        level = 0; fine = []
        if pairs:
            level = 1
            for (b1, b2), dist in pairs.items():
                n1, p1 = precise(b1, b2, ang); n2, p2 = precise(b2, b1, ang)
                if n1 + n2:
                    level = 2; fine.append((cat(b1, b2), n1 + n2, p1 + p2))
        GRID[(p, k, yr, yl)] = level
        rows.append({"hip_pitch_R": p, "knee": k, "yaw_R": yr, "yaw_L": yl, "hull_pairs": {f"{a}|{b}": round(v * 1e3, 2) for (a, b), v in pairs.items()},
                     "hull_categories": sorted({cat(a, b) for a, b in pairs}), "design_interference": fine, "level": level})
n_h = sum(1 for v in GRID.values() if v >= 1); n_d = sum(1 for v in GRID.values() if v == 2)
cats_h = {}; cats_d = {}
for r in rows:
    for c in r["hull_categories"]:
        cats_h[c] = cats_h.get(c, 0) + 1
    for c, _, _ in r["design_interference"]:
        cats_d[c] = cats_d.get(c, 0) + 1
# simple rules: smallest |yaw| difference that touches, at zero pitch / knee
rule = {}
for yr, yl in itertools.product(YAW, YAW):
    if GRID[(0, 0, yr, yl)] == 2:
        rule.setdefault("design_touch_at_zero_pitch_knee", []).append((yr, yl))
# single-leg yaw scan at 1 deg (other leg straight, zero pitch / knee): first contact angle, hull and design
scan = {}
for sgn in (1, -1):
    first_h = first_d = None
    for yv in range(1, 31):
        ang = angles(sgn * yv, 0, 0, 0)
        q = m.qpos0.copy(); set_joints(m, q, {n: math.radians(v) for n, v in ang.items()})
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        prs = {tuple(sorted([body_of_geom(m, c.geom1), body_of_geom(m, c.geom2)])) for c in d.contact[:d.ncon]}
        if prs and first_h is None:
            first_h = (yv * sgn, sorted(prs))
        if prs and first_d is None:
            hit = [(a, b) for a, b in prs if precise(a, b, ang)[0] + precise(b, a, ang)[0]]
            if hit:
                first_d = (yv * sgn, hit)
        if first_h and first_d:
            break
    scan["R_yaw_positive" if sgn > 0 else "R_yaw_negative"] = {"first_hull_contact": first_h, "first_design_contact": first_d}
print("single-leg yaw scan", scan)
out = {"single_leg_yaw_scan": scan, "grid": dict(yaw=YAW, pitch_scissor=PITCH, knee=KNEE), "n_configs": len(GRID), "n_hull_contact": n_h, "n_design_interference": n_d,
       "hull_categories_count": cats_h, "design_categories_count": cats_d, "sample_step_mm": STEP, **rule, "rows": rows}
jdump(out, "test5_leg_interference.json")
print("configs", len(GRID), "hull contact", n_h, "design interference", n_d)
print("hull cats", cats_h); print("design cats", cats_d); print(rule)

plt = korean_font()
from matplotlib.colors import ListedColormap
cm = ListedColormap(["#e8f1fb", "#f5c26b", "#c0392b"])
fig, ax = plt.subplots(len(KNEE), len(PITCH), figsize=(3.0 * len(PITCH), 3.0 * len(KNEE) + 0.8))
for i, k in enumerate(KNEE):
    for j, p in enumerate(PITCH):
        Z = np.array([[GRID[(p, k, yr, yl)] for yr in YAW] for yl in YAW])
        a = ax[i, j]; a.imshow(Z, cmap=cm, vmin=0, vmax=2, origin="lower")
        a.set_xticks(range(len(YAW))); a.set_xticklabels(YAW, fontsize=7); a.set_yticks(range(len(YAW))); a.set_yticklabels(YAW, fontsize=7)
        a.set_title(f"고관절 피치 오른 {p}° / 왼 {-p}°, 무릎 {k}°", fontsize=8)
        if i == len(KNEE) - 1: a.set_xlabel("오른 고관절 요 (°)", fontsize=8)
        if j == 0: a.set_ylabel("왼 고관절 요 (°)", fontsize=8)
import matplotlib.patches as mpatches
fig.legend(handles=[mpatches.Patch(color="#e8f1fb", label="닿지 않음"), mpatches.Patch(color="#f5c26b", label="충돌 껍질만 닿음 (실제 부품은 안 닿음)"),
                    mpatches.Patch(color="#c0392b", label="설계 형상으로도 겹침 (정밀 검사)")], loc="lower center", ncol=3, fontsize=9)
sc = out["single_leg_yaw_scan"]
fig.suptitle(f"한쪽 고관절 요만 {abs(sc['R_yaw_negative']['first_design_contact'][0])}°(뒤꿈치가 안쪽으로) 또는 {abs(sc['R_yaw_positive']['first_design_contact'][0])}°(발끝이 안쪽으로) "
             f"돌려도 두 발이 닿는다 — 격자 {len(GRID)}개 중 껍질 접촉 {n_h}개, 설계 형상 겹침 {n_d}개\n(두 발 평평 유지, 요 범위 ±30°, 시뮬 + 설계 형상 정밀 검사)", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.2); fig.savefig(os.path.join(RES, "test5_leg_interference.png"), dpi=100)
print("saved")
