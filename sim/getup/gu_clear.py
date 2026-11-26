"""팔과 다리 사이 틈(정적): 팔을 늘어뜨린 채(어깨 0°, 팔꿈치 0°) 오른 다리 고관절 롤을 바깥으로 0–20° 벌릴 때 팔 geom 과 다리 geom 사이 가장 짧은 거리.
시뮬은 팔과 다리의 충돌을 계산하지 않으므로(contype 1 / conaffinity 2) 따로 잰다. 다리 geom = 고관절 요 링크 아래 몸(허벅지 · 정강이 · 발목 · 발).
usage: ../.venv/bin/python gu_clear.py → results/clear.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
from gu_model import ARMS, build, RES, jdump


def clearance(name, rolls=(0, 5, 10, 15, 20), arm_deg=0.0):
    m, _ = build(ARMS[name], kp=0); d = mujoco.MjData(m)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    arm_b = {bid("R_arm")} | ({bid("R_forearm")} if name.startswith("팔꿈치") else set())
    leg_b = {bid(n) for n in ("R_thigh", "R_shank", "R_ankle_link", "R_foot")}
    ga = [i for i in range(m.ngeom) if m.geom_bodyid[i] in arm_b and m.geom_contype[i]]
    gl = [i for i in range(m.ngeom) if m.geom_bodyid[i] in leg_b and (m.geom_contype[i] or m.geom_conaffinity[i])]
    ft = np.zeros(6); out = {}
    for r in rolls:
        d.qpos[:] = m.qpos0; d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "R_hip_roll")]] = -math.radians(r)
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "R_arm_swing")]] = math.radians(arm_deg)
        mujoco.mj_forward(m, d)
        out[r] = float(min(mujoco.mj_geomDistance(m, d, a, b, 1.0, ft) for a in ga for b in gl)) * 1e3
    return out


if __name__ == "__main__":
    res = {n: clearance(n) for n in ARMS}
    jdump(res, os.path.join(RES, "clear.json"))
    for n, v in res.items():
        print(n, {k: round(x, 1) for k, x in v.items()})
