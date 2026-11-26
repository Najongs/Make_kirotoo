"""v20 hip roll / pitch ranges on the design solids: collision-free range of the parts below the roll joint
against the hip yaw link + torso (same leg), and the inward roll limit with 3 mm to the other leg (at zero).
Run: cd sim && uv run python tests/hiproll_range_v20.py -> results_v20/hiproll_range.json"""
import os, sys, json, math, numpy as np
from scipy.spatial import cKDTree
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SIM, "..", "code_v17"))
import build20 as B, geom4, build4
from geom4 import rot
links, joints, _ = B.build_model(); J = {j["name"]: j for j in joints}
RES = os.path.join(SIM, "results_v20"); os.makedirs(RES, exist_ok=True)
STEP = 1.0
def pts(ln, skip=()):
    P = [geom4.samples(s, STEP) for pn, ss in links[ln]["parts"].items() if not any(k in pn for k in skip) for s in ss
         if not (s[0] == "cyl" and tuple(s[2]) == tuple(build4.DISC))]
    P = [p for p in P if len(p)]
    return np.vstack(P)
SOL = {ln: [(pn, s, np.array(build4.aabb(build4.sampled(s)))) for pn, ss in L["parts"].items() for s in ss] for ln, L in links.items()}
def inside(ln, Q, M=None):
    if M is not None:
        Mi = np.linalg.inv(M); Q = Q @ Mi[:3, :3].T + Mi[:3, 3]
    hit = {}
    for pn, s, bb in SOL[ln]:
        sel = np.all((Q >= bb[:, 0]) & (Q <= bb[:, 1]), axis=1)
        if sel.any():
            n = int(geom4.inside(s, Q[sel], 0.05).sum())
            if n: hit[pn] = hit.get(pn, 0) + n
    return hit
S = "R"
below = {f"{S}_hip_roll_link": pts(f"{S}_hip_roll_link"), f"{S}_thigh": pts(f"{S}_thigh")}
def M_of(ln, ang):
    chain = {f"{S}_hip_roll_link": ["hip_roll"], f"{S}_thigh": ["hip_roll", "hip_pitch"]}[ln]
    M = np.eye(4)
    for n in chain:
        j = J[f"{S}_{n}"]; M = M @ rot(j["axis"], ang.get(n, 0.0), j["point_mm"])
    return M
def collide(roll, pitch):
    h = {}
    for ln, P in below.items():
        M = M_of(ln, {"hip_roll": roll, "hip_pitch": pitch}); Q = P @ M[:3, :3].T + M[:3, 3]
        for fixed in (f"{S}_hip_yaw_link", "torso"):
            for k, v in inside(fixed, Q).items(): h[f"{ln}->{fixed}:{k}"] = v
        if ln == f"{S}_thigh":       # thigh vs roll link (pitch only)
            Mr = M_of(f"{S}_hip_roll_link", {"hip_roll": roll})
            for k, v in inside(f"{S}_hip_roll_link", Q, Mr).items(): h[f"thigh->roll_link:{k}"] = v
    return h
out = {"roll_free": {}, "pitch_up_free": {}}
for pitch in (-80, -40, 0, 5):
    ok = []
    for roll in range(-40, 41, 1):
        if not collide(roll, pitch): ok.append(roll)
    out["roll_free"][pitch] = [min(ok), max(ok)] if ok else None
    print("pitch", pitch, "roll free", out["roll_free"][pitch], flush=True)
for roll in (-20, 0, 20):
    lim = None
    for pitch in range(0, 31, 1):
        h = collide(roll, pitch)
        if h: lim = (pitch, dict(list(h.items())[:3])); break
    out["pitch_up_free"][roll] = lim
    print("roll", roll, "first pitch-up contact", lim, flush=True)
# inward roll vs the other leg at zero (both thighs + shanks + feet + roll links), 3 mm
other = np.vstack([pts(ln) for ln in ("L_hip_roll_link", "L_thigh", "L_shank", "L_ankle_link", "L_foot", "L_hip_yaw_link")])
T = cKDTree(other)
mine = {ln: pts(ln) for ln in (f"{S}_hip_roll_link", f"{S}_thigh", f"{S}_shank", f"{S}_ankle_link", f"{S}_foot")}
def chainM(ln, roll):
    names = {f"{S}_hip_roll_link": ["hip_roll"], f"{S}_thigh": ["hip_roll", "hip_pitch"], f"{S}_shank": ["hip_roll", "hip_pitch", "knee"],
             f"{S}_ankle_link": ["hip_roll", "hip_pitch", "knee", "ankle_pitch"], f"{S}_foot": ["hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]}[ln]
    M = np.eye(4)
    for n in names:
        j = J[f"{S}_{n}"]; M = M @ rot(j["axis"], roll if n == "hip_roll" else 0.0, j["point_mm"])
    return M
gaps = {}
for roll in range(0, 41, 1):          # + = inward for the right leg
    g = min(float(T.query(P @ chainM(ln, roll)[:3, :3].T + chainM(ln, roll)[:3, 3], k=1, distance_upper_bound=30)[0].min()) for ln, P in mine.items())
    gaps[roll] = g
out["inward_gap_vs_other_leg_mm"] = gaps
lim = next((r for r in range(0, 41) if gaps[r] < 3.0), None)
out["inward_limit_3mm_deg"] = (lim - 1) if lim else ">40"
print("inward 3 mm limit", out["inward_limit_3mm_deg"], "(sampling step 1 mm: over-estimates the gap by <= 1.7 mm)")
json.dump(out, open(os.path.join(RES, "hiproll_range.json"), "w"), indent=1)
# hip region only (roll link + thigh vs the other leg's roll link + thigh + yaw link), as in HIPROLL_STUDY
other_h = np.vstack([pts(ln) for ln in ("L_hip_roll_link", "L_thigh", "L_hip_yaw_link")])
Th = cKDTree(other_h)
gh = {}
for roll in range(0, 41, 1):
    gh[roll] = min(float(Th.query(P @ chainM(ln, roll)[:3, :3].T + chainM(ln, roll)[:3, 3], k=1, distance_upper_bound=40)[0].min())
                   for ln, P in mine.items() if ln in (f"{S}_hip_roll_link", f"{S}_thigh"))
out["inward_gap_hip_region_mm"] = gh
limh = next((r for r in range(0, 41) if gh[r] < 3.0), None)
out["inward_limit_3mm_hip_region_deg"] = (limh - 1) if limh else ">40"
print("hip-region inward 3 mm limit", out["inward_limit_3mm_hip_region_deg"])
json.dump(out, open(os.path.join(RES, "hiproll_range.json"), "w"), indent=1)
