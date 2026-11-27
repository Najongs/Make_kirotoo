"""v21 render module for sim/export_mjcf.py (KITORU_VER=21): build21_arm geometry (v21 head nod -20 .. +28 +
elbow arm), render6 chains as in render21_arm.py without its whole-body 'base' joint, nod_pose from head21.

The torso / head print fits of build21_body (0.1 .. 0.3 mm) are not in here: they do not change the sim meshes, and
out21_arm/mass_build21_arm_light.json was computed from build21_arm as well.
"""
import os, sys, math
import numpy as np
os.environ["BUILD"] = "build21_arm"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render6 as r
import head21 as H21

r.CHAINS.pop("neck", None); r.CHAINS["head"] = ["head_nod"]
for k in ("arm_crank", "R_arm_rod", "L_arm_rod"):
    r.CHAINS.pop(k, None)
r.CHAINS.update({"R_arm": ["R_arm_swing"], "L_arm": ["L_arm_swing"], "R_forearm": ["R_arm_swing", "R_elbow"],
                 "L_forearm": ["L_arm_swing", "L_elbow"], "nod_crank": ["head_nod", "nod_crank"], "nod_rod": ["nod_rod"],
                 "ears": ["head_nod", "ear_axle"], "ear_rod": ["head_nod", "nod_crank", "ear_rod"]})
for S_ in "RL":     # v20 leg chain: yaw -> roll -> pitch -> knee -> ankle pitch -> ankle roll
    nm = [f"{S_}_hip_yaw", f"{S_}_hip_roll", f"{S_}_hip_pitch", f"{S_}_knee", f"{S_}_ankle_pitch", f"{S_}_ankle_roll"]
    for i, ln in enumerate([f"{S_}_hip_yaw_link", f"{S_}_hip_roll_link", f"{S_}_thigh", f"{S_}_shank", f"{S_}_ankle_link", f"{S_}_foot"]):
        r.CHAINS[ln] = nm[:i + 1]
missing = set(r.links) - set(r.CHAINS)
assert not missing, missing


def nod_pose(beta):
    """closed loop (head21 = the v21 loop solver): head nod beta deg -> crank, ear axle, rods (deg)"""
    al = H21.nod(beta)["alpha"]; ps = H21.ear(al)["psi"]
    pw = H21.rot2(H21.rot2(H21.S + np.array([H21.NOD_R, 0.0]), al, H21.S), beta, H21.P)
    A_ = np.array(H21.NOD_A); pin0 = H21.S + np.array([H21.NOD_R, 0.0])
    d = -math.degrees(math.atan2(*(pw - A_)[::-1]) - math.atan2(*(pin0 - A_)[::-1]))
    P1, P2, LE = H21.ear_geom()
    q1 = H21.rot2(P1, al, H21.S); q2 = H21.rot2(P2, ps, H21.E0)
    ang = lambda u: math.degrees(math.atan2(u[1], u[0]))
    rod = -(ang(q2 - q1) - ang(P2 - P1))
    return {"head_nod": beta, "nod_crank": al, "nod_rod": d, "ear_axle": ps, "ear_rod": rod - al}
