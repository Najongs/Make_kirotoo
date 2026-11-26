"""Per-crouch-depth yaw limit (first angle with a leg-leg gap < 3 mm, coarse 1.0 mm sampling) for a build18 cut.
usage: KITORU_THETA=20 python tests/foot_clearance_by_crouch.py"""
import json
import sys, math; sys.path.insert(0,'tests')
import os; os.environ.setdefault("KITORU_THETA", "20")
from foot_clearance import Clearance
import servo_study as S
C = Clearance("build18", step=1.0)
OUT = {}
for c in (0, 15, 30, 40, 45):
    q1, q2 = S.leg_ik(0.0, -(150.0 - c)); q1, q2 = math.degrees(q1), math.degrees(q2)
    res = []
    for th in list(range(-31, -14)) + list(range(14, 32)):
        g, w = C.gap_crouch(th, q1, q2)
        res.append((th, round(g, 2), w[0][2:] + "-" + w[2][2:] if w else ""))
    lo = max([t for t, g, _ in res if t < 0 and g < 3.0], default=None); hi = min([t for t, g, _ in res if t > 0 and g < 3.0], default=None)
    OUT[c] = {"first_below_3mm_heel_in": lo, "first_below_3mm_toe_in": hi, "gaps": res}
    print(f"crouch {c}: first <3mm heel-in {lo}, toe-in {hi}", [r for r in res if r[0] in (-21, -20, 18, 19, 20, 21, 22, 24)], flush=True)
json.dump(OUT, open(f"results_v18/clearance_by_crouch_t{os.environ['KITORU_THETA']}.json", "w"), indent=1)
