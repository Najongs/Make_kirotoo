import os, math, numpy as np
os.environ["BUILD"] = "build12"
import render6 as r
from scipy.optimize import brentq
B = r.build5
r.CHAINS.pop("neck", None); r.CHAINS["head"] = ["head_nod"]
r.CHAINS.update({"R_arm": ["R_arm_swing"], "L_arm": ["L_arm_swing"], "arm_crank": ["arm_crank"],
                 "R_arm_rod": ["arm_crank", "R_arm_rod"], "L_arm_rod": ["arm_crank", "L_arm_rod"],
                 "nod_crank": ["head_nod", "nod_crank"], "nod_rod": ["nod_rod"]})
P_ = np.array([0.0, B.ZT9]); S0 = np.array(B.NOD_S); A_ = np.array(B.NOD_A)
pin0 = S0 + np.array([B.NOD_R, 0.0]); elln = np.hypot(*(pin0 - A_))
def rot2(v, deg, c):
    a = math.radians(deg); d = v - c
    return c + np.array([d[0] * math.cos(a) + d[1] * math.sin(a), -d[0] * math.sin(a) + d[1] * math.cos(a)])
def nod_pose(beta):
    al = brentq(lambda a: np.hypot(*(rot2(rot2(pin0, a, S0), beta, P_) - A_)) - elln, -45, 70)
    pw = rot2(rot2(pin0, al, S0), beta, P_)
    d = -math.degrees(math.atan2(*(pw - A_)[::-1]) - math.atan2(*(pin0 - A_)[::-1]))
    return {"head_nod": beta, "nod_crank": al, "nod_rod": d}
C = np.array(B.CRANK); S = np.array([0.0, B.SH_Z])
def arm_pose(th_deg):
    th = math.radians(th_deg); out = {"arm_crank": -th_deg}
    for nm, rc, L, sg in (("R", B.R_CRANK, B.R_LEV, 1), ("L", B.L_CRANK, B.L_LEV, -1)):
        P0 = C + sg * np.array([rc, 0.0]); Pp = C + sg * rc * np.array([math.cos(th), math.sin(th)])
        T0 = S + np.array([L, 0.0]); ell = np.hypot(*(T0 - P0))
        ph = brentq(lambda p: np.hypot(*(S + L * np.array([math.cos(p), math.sin(p)]) - Pp)) - ell, -0.8, 0.8)
        Tp = S + L * np.array([math.cos(ph), math.sin(ph)])
        d = math.degrees(math.atan2(*(Tp - Pp)[::-1]) - math.atan2(*(T0 - P0)[::-1]))
        out[f"{nm}_arm_swing"] = -math.degrees(ph)
        out[f"{nm}_arm_rod"] = th_deg - d
    return out
walk = { "R_hip_pitch": -28, "R_knee": 40, "R_ankle_pitch": -12,
        "L_hip_pitch": 12, "L_knee": 8, "L_ankle_pitch": -20}
walk.update(arm_pose(-40)); walk.update(nod_pose(10))
r.panel([r.render(15, -35), r.render(0, 0), r.render(0, -90), r.render(12, -55, walk)],
        "/home/claude/biped/out12/biped_v12_preview.png")
inside = dict(arm_pose(-40))
r.panel([r.render(20, -40, inside, skip=("torso_cover", "lid", "thigh_shell", "shank_shell", "shorts")),
         r.render(5, -90, inside, skip=("torso_cover", "lid", "thigh_shell", "shank_shell", "shorts", "electronics", "jetson", "opencr", "battery", "dcdc", "imu", "shelf")),
         r.render(10, -70, nod_pose(20), skip=("torso_cover", "shorts", "head_front", "face_plate", "eye_", "mouth", "antenna", "crest", "camera"))],
        "/home/claude/biped/out12/biped_v12_inside.png")
print("ok")
