"""걷기용 접은 팔과 다리 사이 틈(정적 계산): 어깨 a°, 팔꿈치 e°(− = 아래팔을 앞으로)로 접은 채 오른 다리를
고관절 롤 바깥 0–12°(걷기 중 실측 최대) · 20° · 33.6°(형상 한계) × 고관절 피치 −40 · 0 · +20° 로 움직일 때
팔(위팔 · 아래팔 · 손) ↔ 다리(요 링크 아래 허벅지 · 정강이 · 발목 · 발) 가장 짧은 거리.
usage: ../.venv/bin/python gu_clear2.py → results/clear2.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
from gu_model import ARMS, build, RES, jdump

ROLLS = (0, 4, 8, 12, 20, 33.6)
PITCH = (-40, 0, 20)


def clear(name, a_deg, e_deg):
    m, _ = build(ARMS[name], kp=0, hip_min=-2.06, knee_max=2.0); d = mujoco.MjData(m)
    J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    nm = lambda i: m.body(m.geom_bodyid[i]).name
    ga = [i for i in range(m.ngeom) if nm(i) in ("R_arm", "R_forearm") and m.geom_contype[i]]
    gl = [i for i in range(m.ngeom) if nm(i) in ("R_hip_yaw_link", "R_thigh", "R_shank", "R_ankle_link", "R_foot") and (m.geom_contype[i] or m.geom_conaffinity[i])]
    ft = np.zeros(6); out = {}
    for r in ROLLS:
        dd = []
        for p in PITCH:
            d.qpos[:] = m.qpos0; d.qpos[J("R_hip_roll")] = -math.radians(r); d.qpos[J("R_hip_pitch")] = math.radians(p)
            d.qpos[J("R_arm_swing")] = math.radians(a_deg)
            if "R_forearm" in [m.body(i).name for i in range(m.nbody)]:
                d.qpos[J("R_elbow")] = math.radians(e_deg)
            mujoco.mj_forward(m, d)
            dd.append(min(mujoco.mj_geomDistance(m, d, x, y, 0.5, ft) for x in ga for y in gl) * 1e3)
        out[r] = float(min(dd))
    return out


if __name__ == "__main__":
    res = {}
    for n in ("팔꿈치 100+100", "팔꿈치 110+110", "팔꿈치 120+120"):
        for a, e in ((0, 0), (0, -45), (0, -60), (0, -90), (-15, -90), (10, -100)):
            res[f"{n} / 어깨 {a}° 팔꿈치 {e}°"] = clear(n, a, e)
    jdump(res, os.path.join(RES, "clear2.json"))
    for k, v in res.items():
        print(k, {r: round(x, 1) for r, x in v.items()})
