"""two-leg hip roll map (design solids, layout A or B): smallest gap between the right and left leg (roll link down to
the foot) for right roll a and left roll b (other joints 0); 3 mm rule.  Right + = inward, left - = inward.
Run: cd sim && uv run python tests/roll_pair_v20.py [tag] -> results_v20/roll_pair_<tag>.json"""
import os, sys, json, numpy as np
from scipy.spatial import cKDTree
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SIM, "..", "code_v17"))
import build20 as B, geom4, build4
from geom4 import rot
tag = sys.argv[1] if len(sys.argv) > 1 else B.LAYOUT
links, joints, _ = B.build_model(); J = {j["name"]: j for j in joints}
RES = os.path.join(SIM, "results_v20"); os.makedirs(RES, exist_ok=True)
LN = ("hip_roll_link", "thigh", "shank", "ankle_link", "foot")
def pts(ln):
    P = [geom4.samples(s, 1.5) for ss in links[ln]["parts"].values() for s in ss if not (s[0] == "cyl" and tuple(s[2]) == tuple(build4.DISC))]
    return np.vstack([p for p in P if len(p)])
P = {S: np.vstack([pts(f"{S}_{k}") for k in LN]) for S in "RL"}
def leg(S, a):
    j = J[f"{S}_hip_roll"]; M = rot(j["axis"], a, j["point_mm"])
    return P[S] @ M[:3, :3].T + M[:3, 3]
angles = np.arange(-25, 25.1, 2.5)
G = np.zeros((len(angles), len(angles)))
TL = {b: cKDTree(leg("L", b)) for b in angles}
for i, a in enumerate(angles):
    Q = leg("R", a)
    for k, b in enumerate(angles):
        G[i, k] = float(TL[b].query(Q, k=1, distance_upper_bound=60)[0].min())
def lim(f):
    ok = [x for x in np.arange(0, 25.1, 0.5) if f(x) >= 3.0]
    last = 0.0
    for x in np.arange(0, 25.1, 0.5):
        if f(x) < 3.0: break
        last = x
    return last
def gap(a, b):
    return float(cKDTree(leg("L", b)).query(leg("R", a), k=1, distance_upper_bound=60)[0].min())
out = dict(layout=B.LAYOUT, angles=angles.tolist(), gap_mm=G.round(2).tolist(),
           inward_other_zero_deg=lim(lambda x: gap(x, 0.0)),
           both_inward_deg=lim(lambda x: gap(x, -x)),
           sway_same_direction_deg=lim(lambda x: gap(x, x)),
           note="right + = inward; left - = inward; sway = both legs tilt the same way (right +x, left +x)")
json.dump(out, open(os.path.join(RES, f"roll_pair_{tag}.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k.endswith("_deg") or k == "layout"})
