"""Fine re-check of the closest leg-leg spots found by foot_clearance.py: sample only the parts near the contact
point at a finer spacing (default 0.25 mm) so the sampling over-estimate drops to <= 0.43 mm.
usage: KITORU_THETA=20 python tests/foot_clearance_fine.py build18"""
import sys, math, os, json
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foot_clearance import Clearance, LINKS
import geom4, build4
import servo_study as S

mod = sys.argv[1] if len(sys.argv) > 1 else "build18"
STEP = 0.25
C = Clearance(mod, step=1.0)
cases = []
for c in (0.0, 15.0, 30.0, 40.0):
    q1, q2 = S.leg_ik(0.0, -(150.0 - c)); cases.append((c, math.degrees(q1), math.degrees(q2)))
out = []
FORCE = os.environ.get("FINE_PAIR")       # e.g. "ankle_link:R_ankle_bracket:L_ankle_bracket" -> check that pair only
THETAS = [int(t) for t in os.environ.get("FINE_THETAS", "-21,-20,20,21").split(",")]
for th in THETAS:
    for c, q1, q2 in cases:
        if FORCE:
            ln, pa_, pb_ = FORCE.split(":")
            aR0 = {"hip_yaw": th, "hip_pitch": q1, "knee": q2, "ankle_pitch": -(q1 + q2)}
            aL0 = {"hip_yaw": 0.0, "hip_pitch": q1, "knee": q2, "ankle_pitch": -(q1 + q2)}
            MR0, ML0 = C.pose("R", aR0)[ln], C.pose("L", aL0)[ln]
            PA = C.P[f"R_{ln}"][C.lab[f"R_{ln}"] == pa_] @ MR0[:3, :3].T + MR0[:3, 3]
            PB = C.P[f"L_{ln}"][C.lab[f"L_{ln}"] == pb_] @ ML0[:3, :3].T + ML0[:3, 3]
            dd, ii = cKDTree(PB).query(PA, k=1); j = int(np.argmin(dd))
            g, w = float(dd[j]), (f"R_{ln}", pa_, f"L_{ln}", pb_, list(PA[j]))
        else:
            g, w = C.gap_crouch(th, q1, q2)
        if w is None or g > 6.0:
            continue
        la, pa, lb, pb, xyz = w
        aR = {"hip_yaw": th, "hip_pitch": q1, "knee": q2, "ankle_pitch": -(q1 + q2)}
        aL = {"hip_yaw": 0.0, "hip_pitch": q1, "knee": q2, "ankle_pitch": -(q1 + q2)}
        MR, ML = C.pose("R", aR)[la[2:]], C.pose("L", aL)[lb[2:]]
        X = np.array(xyz)
        def near(ln, part, M):
            pts = []
            for s in C.links[ln]["parts"][part]:
                if s[0] == "cyl" and tuple(s[2]) == tuple(build4.DISC):
                    continue
                b = build4.aabb(build4.sampled(s))
                # local box around the contact point
                Xl = np.linalg.inv(M) @ np.r_[X, 1.0]
                lo = np.maximum([x[0] for x in b], Xl[:3] - 15); hi = np.minimum([x[1] for x in b], Xl[:3] + 15)
                if np.any(lo >= hi):
                    continue
                ax = [np.arange(l + STEP / 2, h, STEP) for l, h in zip(lo, hi)]
                G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
                G = G[geom4.inside(s, G)]
                pts.append(G @ M[:3, :3].T + M[:3, 3])
            return np.concatenate(pts) if pts else np.zeros((0, 3))
        A, B = near(la, pa, MR), near(lb, pb, ML)
        gf = float(cKDTree(B).query(A, k=1)[0].min()) if len(A) and len(B) else None
        out.append({"theta": th, "crouch_mm": c, "coarse_gap": g, "fine_gap": gf, "pair": [pa, pb], "fine_error_mm": STEP * math.sqrt(3)})
        print(th, c, round(g, 2), None if gf is None else round(gf, 2), pa, pb, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v18" if mod == "build18" else "results_v19",
                                 f"clearance_fine_{os.environ.get('KITORU_THETA', 'x')}_{(FORCE or 'closest').split(':')[0]}.json"), "w"), indent=1)
