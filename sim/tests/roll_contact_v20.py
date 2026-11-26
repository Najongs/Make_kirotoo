"""where the two legs meet in inward hip roll (design solids): nearest point pair (part, world point) between the right
leg (rolled a) and the left leg (rolled b), other joints 0 or with hip yaw.  Run: cd sim && uv run python tests/roll_contact_v20.py"""
import os, sys, json, numpy as np
from scipy.spatial import cKDTree
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SIM, "..", "code_v17"))
import build20 as B, geom4, build4
from geom4 import rot
links, joints, _ = B.build_model(); J = {j["name"]: j for j in joints}
LN = ("hip_roll_link", "thigh", "shank", "ankle_link", "foot")
def pts(S):
    P, lab = [], []
    for k in LN:
        for pn, ss in links[f"{S}_{k}"]["parts"].items():
            for s in ss:
                if s[0] == "cyl" and tuple(s[2]) == tuple(build4.DISC): continue
                q = geom4.samples(s, 1.0)
                if len(q): P.append(q); lab += [f"{S}_{k}:{pn}"] * len(q)
    return np.vstack(P), np.array(lab)
P = {S: pts(S) for S in "RL"}
def leg(S, roll, yaw=0.0):
    jr = J[f"{S}_hip_roll"]; jy = J[f"{S}_hip_yaw"]
    M = rot(jy["axis"], yaw, jy["point_mm"]) @ rot(jr["axis"], roll, jr["point_mm"])
    return P[S][0] @ M[:3, :3].T + M[:3, 3]
def near(a, b, yr=0.0, yl=0.0):
    QR, QL = leg("R", a, yr), leg("L", b, yl)
    d, i = cKDTree(QL).query(QR, k=1)
    j = int(np.argmin(d))
    return float(d[j]), P["R"][1][j], QR[j].round(1).tolist(), P["L"][1][i[j]], QL[i[j]].round(1).tolist()
def lim(f):
    last = 0.0
    for x in np.arange(0, 30.01, 0.25):
        if f(x)[0] < 3.0: break
        last = x
    return last, f(last + 0.25)
out = {}
for nm, f in (("other_zero", lambda x: near(x, 0.0)), ("both_inward", lambda x: near(x, -x)),
              ("other_zero_yaw+25", lambda x: near(x, 0.0, 25.0, 0.0)), ("other_zero_yaw-25", lambda x: near(x, 0.0, -25.0, 0.0)),
              ("both_inward_yaw_opposite25", lambda x: near(x, -x, 25.0, -25.0))):
    l, c = lim(f); out[nm] = dict(limit_deg=l, first_contact=c[1:]); print(nm, l, c, flush=True)
json.dump(out, open(os.path.join(SIM, "results_v20", f"roll_contact_{sys.argv[1] if len(sys.argv) > 1 else 'now'}.json"), "w"), indent=1)
