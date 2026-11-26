"""(무게중심 목표: 디딤발 가운데 위, 또는 안쪽 20 mm)
흔드는 발을 옆으로 얼마나 옮길 수 있나(운동학만): 오른발을 디딤발로 고정, 몸통 높이 · 무게중심을 디딤발 위로 두고
왼발(흔드는 발, 바닥에서 12 mm 들고 평평)을 기준 자리(y = +80 mm)에서 옆으로 dy 옮기라고 역기구학에 시킨 뒤 실제로 간 거리.
python lateral_reach.py → results/lateral_reach.json"""
import os, numpy as np, mujoco
from kw_common import build_model, RES, jdump
from kw_ctrl import WholeBodyIK

out = {}
for name, hr, cy in (("기준(고관절 롤 없음)", False, 0.0), ("고관절 롤 추가", True, 0.0), ("고관절 롤 추가, 무게중심 안쪽 20 mm", True, 0.02)):
    m, _ = build_model(hip_roll=hr)
    ik = WholeBodyIK(m); ix = ik.ix
    q0 = m.qpos0.copy(); q0[3:7] = [1, 0, 0, 0]
    for s in "RL":
        q0[ix.qadr[f"{s}_hip_pitch"]] = -0.45; q0[ix.qadr[f"{s}_knee"]] = 0.9; q0[ix.qadr[f"{s}_ankle_pitch"]] = -0.45
    d = ik.d; d.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, d)
    fR = d.xpos[ix.foot["R"]].copy(); fL = d.xpos[ix.foot["L"]].copy()
    rows = []
    for dy in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04):
        q = q0.copy()
        tgt = dict(feet={"R": (fR, 0.0, np.full(3, 100.0), 100.0),
                         "L": (fL + [0, dy, 0.012], 0.0, np.full(3, 100.0), 100.0)},
                   com=fR[:2] + [-0.007, cy], wcom=10.0, pitch=0.0, yaw=0.0, wori=10.0, z=-0.025, wz=3.0)
        q = ik.solve(q, tgt, iters=300)
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        got = d.xpos[ix.foot["L"]] - d.xpos[ix.foot["R"]]
        R = d.xmat[ix.foot["L"]].reshape(3, 3)
        rows.append(dict(dy_cmd_mm=dy * 1e3, dy_got_mm=(got[1] - (fL - fR)[1]) * 1e3, foot_tilt_deg=float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
                         com_err_mm=float(np.linalg.norm(d.subtree_com[ix.torso][:2] - tgt["com"]) * 1e3),
                         torso_roll_deg=float(np.degrees(np.arctan2(d.xmat[ix.torso][7], d.xmat[ix.torso][8])))))
    out[name] = rows
    print(name)
    for r in rows:
        print("  ", {k: round(v, 1) for k, v in r.items()})
jdump(out, os.path.join(RES, "lateral_reach.json"))
