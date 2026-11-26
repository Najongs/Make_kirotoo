import os, math, numpy as np
os.environ["BUILD"] = "build16"
import render6 as r
from scipy.optimize import brentq
B = r.build5
import build16 as B16
r.CHAINS.pop("neck", None); r.CHAINS["head"] = ["head_nod"]
for k in ("arm_crank", "R_arm_rod", "L_arm_rod"):
    r.CHAINS.pop(k, None)
r.CHAINS.update({"R_arm": ["R_arm_swing"], "L_arm": ["L_arm_swing"],
                 "nod_crank": ["head_nod", "nod_crank"], "nod_rod": ["nod_rod"],
                 "ears": ["head_nod", "ear_axle"], "ear_rod": ["head_nod", "nod_crank", "ear_rod"]})
P_ = np.array([0.0, B.ZT9]); S0 = np.array(B.NOD_S); A_ = np.array(B.NOD_A)
pin0 = S0 + np.array([B.NOD_R, 0.0]); elln = np.hypot(*(pin0 - A_))
E0 = np.array(B16.EAR_C); P1, P2, LE = B16.ear_geometry(); P1 = np.array(P1); P2 = np.array(P2)
def rot2(v, deg, c):
    a = math.radians(deg); d = v - c
    return c + np.array([d[0] * math.cos(a) + d[1] * math.sin(a), -d[0] * math.sin(a) + d[1] * math.cos(a)])
def ang(u): return math.degrees(math.atan2(u[1], u[0]))
def ear_pose(al):
    q1 = rot2(P1, al, S0)
    ps = brentq(lambda p: np.hypot(*(rot2(P2, p, E0) - q1)) - LE, -89, 89)
    q2 = rot2(P2, ps, E0)
    rod = -(ang(q2 - q1) - ang(P2 - P1))
    return {"ear_axle": ps, "ear_rod": rod - al}
def nod_pose(beta):
    al = brentq(lambda a: np.hypot(*(rot2(rot2(pin0, a, S0), beta, P_) - A_)) - elln, -45, 70)
    pw = rot2(rot2(pin0, al, S0), beta, P_)
    d = -math.degrees(math.atan2(*(pw - A_)[::-1]) - math.atan2(*(pin0 - A_)[::-1]))
    out = {"head_nod": beta, "nod_crank": al, "nod_rod": d}
    out.update(ear_pose(al))
    return out
if __name__ == "__main__":
    walk = {"R_hip_pitch": -28, "R_knee": 40, "R_ankle_pitch": -12, "L_hip_pitch": 12, "L_knee": 8, "L_ankle_pitch": -20,
            "R_arm_swing": 25, "L_arm_swing": -25}
    walk.update(nod_pose(15))
    r.panel([r.render(15, -35), r.render(0, 0), r.render(0, -90), r.render(12, -55, walk)],
            "/home/claude/biped/out16/biped_v16_preview.png")
    up = {"R_arm_swing": -100, "L_arm_swing": -100}; up.update(nod_pose(-10))
    r.panel([r.render(20, -40, {}, skip=("torso_cover", "lid", "thigh_shell", "shank_shell", "shorts", "arm_R_shell", "arm_L_shell", "hand")),
             r.render(-35, -30, {}, skip=()),
             r.render(10, -60, up)],
            "/home/claude/biped/out16/biped_v16_inside.png")
    keep = ("head_back", "head_nod_servo", "nod_crank", "nod_rod", "neck_yoke", "neck_shaft", "neck_bearings", "nod_anchor",
            "ear_", "ear_rod", "nod_ear")
    allg = r.GEOM[:]
    r.GEOM[:] = [g for g in allg if any(k in g[1] for k in keep) and "antenna" not in g[1]]
    r.panel([r.render(15, 30, nod_pose(0), scale=3.0), r.render(15, 30, nod_pose(20), scale=3.0)],
            "/home/claude/biped/out16/biped_v16_head_inside.png")
    r.GEOM[:] = allg
    for b in (-10, 0, 10, 20):
        p = nod_pose(b); print("nod", b, "crank", round(p["nod_crank"], 1), "ear", round(p["ear_axle"], 1))
    print("ok")
