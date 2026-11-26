import os, math, numpy as np
os.environ["BUILD"] = "build10"
import render6 as r
from scipy.optimize import brentq
B = r.build5
r.CHAINS.pop("neck", None); r.CHAINS["head"] = ["head_nod"]
r.CHAINS.update({"R_arm": ["R_arm_swing"], "L_arm": ["L_arm_swing"], "arm_crank": ["arm_crank"],
                 "R_arm_rod": ["arm_crank", "R_arm_rod"], "L_arm_rod": ["arm_crank", "L_arm_rod"]})
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
walk = {"head_nod": 10, "R_hip_pitch": -28, "R_knee": 40, "R_ankle_pitch": -12,
        "L_hip_pitch": 12, "L_knee": 8, "L_ankle_pitch": -20}
walk.update(arm_pose(-40))
r.panel([r.render(15, -35), r.render(0, 0), r.render(0, -90), r.render(12, -55, walk)],
        "/home/claude/biped/out10/biped_v10_preview.png")
inside = dict(arm_pose(-40))
r.panel([r.render(20, -40, inside, skip=("torso_cover", "lid", "thigh_shell", "shank_shell", "shorts")),
         r.render(5, -90, inside, skip=("torso_cover", "lid", "thigh_shell", "shank_shell", "shorts", "electronics", "jetson", "opencr", "battery", "dcdc", "imu", "shelf")),
         r.render(20, 40, {"head_nod": 20}, skip=("torso_cover", "shorts"))],
        "/home/claude/biped/out10/biped_v10_inside.png")
print("ok")
