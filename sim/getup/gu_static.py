"""정적 계산: 「손으로 바닥을 짚은 채, 무게중심이 발바닥 위에 오는」 자세에서 어깨가 바닥에서 얼마나 높은가.
이 높이가 손에서 발로 무게가 넘어가는 순간에 팔이 닿아야 하는 길이(준정적 일어나기에 필요한 팔 길이)다.

자세: 시상면 대칭, 두 발 평평(몸 전체를 y 축으로 돌려 맞춤), 무릎 K · 발목 A · 고관절 피치 H 를 격자로 훑음.
팔은 세계 기준 수직 아래(손이 어깨 바로 밑)로 둔다(팔 질량도 무게중심에 넣음).
조건: 무게중심 x 가 패드 앞뒤 끝 안쪽 5 mm 안. 이 조건을 만족하는 자세 중 어깨 축 높이(발바닥 기준)가 가장 낮은 것을 몸통이 앞으로 기운 자세(front, 엎드림에서 오는 길) · 뒤로 기운 자세(back, 누움에서 오는 길)로 나눠 찾는다.
관절 범위: 무릎 0–Kmax(설계 100°, 민감도 110 · 120 · 130°), 발목 −50 – +22°, 고관절 −80 – +30°(설계).
usage: cd sim/getup && ../.venv/bin/python gu_static.py → results/static.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
from gu_model import build, ARMS, reach, arm_mass_g, RES, jdump


def low_shoulder(arm, kmax_deg, margin=0.005, hip_min=-1.396):
    m, _ = build(arm, knee_max=math.radians(max(kmax_deg, 100)), hip_min=hip_min, kp=0)
    d = mujoco.MjData(m)
    q = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    fR, tor, armR = B("R_foot"), B("torso"), B("R_arm")
    pads = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("R_foot_pad")]
    best = {}
    for K in np.radians(np.arange(0, kmax_deg + 0.1, 5)):
        for A in np.radians(np.arange(-50, 22.1, 4)):
            for H in np.radians(np.arange(math.degrees(hip_min), 30.1, 3)):
                d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[:3] = 0
                for s in "RL":
                    d.qpos[q(f"{s}_hip_pitch")] = H; d.qpos[q(f"{s}_knee")] = K; d.qpos[q(f"{s}_ankle_pitch")] = A
                mujoco.mj_kinematics(m, d)
                Rf = d.xmat[fR].reshape(3, 3); ang = math.atan2(Rf[0, 2], Rf[2, 2])
                qq = np.zeros(4); mujoco.mju_axisAngle2Quat(qq, np.array([0, 1.0, 0]), -ang); d.qpos[3:7] = qq
                mujoco.mj_kinematics(m, d)
                v = d.xmat[tor].reshape(3, 3).T @ np.array([0, 0, -1.0]); th = math.atan2(-v[0], -v[2])
                for s in "RL":
                    d.qpos[q(f"{s}_arm_swing")] = th
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                px = [d.geom_xpos[g][0] for g in pads]; pr = m.geom_size[pads[0]][0]
                sole = min(d.geom_xpos[g][2] for g in pads) - m.geom_size[pads[0]][1]
                cx = d.subtree_com[1][0]
                if not (min(px) - pr + margin <= cx <= max(px) + pr - margin):
                    continue
                hs = d.xpos[armR][2] - sole
                # 손이 발 · 정강이와 겹치지 않게: 어깨 x 가 발끝보다 앞이거나 발뒤꿈치보다 뒤(손을 발 옆에 짚는 것은 허용 — 팔은 다리 바깥 y 에 있음)
                zt = d.xmat[tor].reshape(3, 3)[:, 2]; lean = math.degrees(math.atan2(zt[0], zt[2]))     # 몸통 앞 기울기(+ 앞으로)
                side = "front" if lean >= 0 else "back"
                if side not in best or hs < best[side]["shoulder_h_mm"] / 1e3:
                    best[side] = dict(shoulder_h_mm=hs * 1e3, K_deg=math.degrees(K), A_deg=math.degrees(A), H_deg=math.degrees(H),
                                arm_deg=math.degrees(th), torso_lean_deg=lean, com_x_rel_mm=(cx - np.mean(px)) * 1e3,
                                shoulder_x_rel_mm=(d.xpos[armR][0] - np.mean(px)) * 1e3)
    return best


if __name__ == "__main__":
    out = []
    for name in ("곧은 131 (지금)", "곧은 220", "팔꿈치 110+110"):
        for km in (100, 110, 120, 130):
            bb = low_shoulder(ARMS[name], km)
            r = reach(ARMS[name])
            for side, b in bb.items():
                row = dict(arm=name, reach_mm=r * 1e3, arm_g=arm_mass_g(ARMS[name]), knee_max_deg=km, side=side, **b)
                row["gap_mm"] = b["shoulder_h_mm"] - r * 1e3
                out.append(row); print({k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}, flush=True)
    jdump(out, os.path.join(RES, "static.json"))
