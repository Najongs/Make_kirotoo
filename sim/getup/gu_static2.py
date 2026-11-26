"""정적 계산 2 (안 A 최종 형상 범위): 고관절 피치 −115°(형상 한계 −117.8° 에서 여유 3°) · 무릎 115°(한계 120.1°).
gu_static.py 와 같은 방법(두 발 평평, 무게중심이 패드 앞뒤 끝 안쪽 5 mm 안, 팔은 세계 수직 아래)에,
다리(허벅지 · 정강이 · 발목 · 발) ↔ 몸통 · 머리 메시 거리가 0 이상인 자세만 받는다(시뮬 모델은 v18 메시 + 안 A 근사라 실제 v20A 와 다를 수 있음).
usage: ../.venv/bin/python gu_static2.py → results/static2.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
from gu_model import build, ARMS, RES, jdump

LEGB = ("R_thigh", "L_thigh", "R_shank", "L_shank", "R_ankle_link", "L_ankle_link", "R_foot", "L_foot")


def low_shoulder2(arm, kmax_deg, hip_min_deg, margin=0.005, clear=0.0):
    m, _ = build(arm, knee_max=math.radians(max(kmax_deg, 100)), hip_min=math.radians(min(hip_min_deg, -80)), kp=0)
    d = mujoco.MjData(m); ft = np.zeros(6)
    q = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    fR, tor, armR = B("R_foot"), B("torso"), B("R_arm")
    nm = lambda i: m.body(m.geom_bodyid[i]).name
    skip = lambda i: any(k in (m.geom(i).name or "") for k in ("arm_", "shoulder_shaft"))
    legg = [i for i in range(m.ngeom) if nm(i) in LEGB and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[i] or m.geom_conaffinity[i])]
    bodyg = [i for i in range(m.ngeom) if nm(i) in ("torso", "head") and m.geom_contype[i] and not skip(i)]
    pads = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("R_foot_pad")]
    best = {}
    for K in np.radians(np.arange(0, kmax_deg + 0.1, 5)):
        for A in np.radians(np.arange(-50, 22.1, 4)):
            for H in np.radians(np.arange(hip_min_deg, 30.1, 3)):
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
                zt = d.xmat[tor].reshape(3, 3)[:, 2]; lean = math.degrees(math.atan2(zt[0], zt[2]))
                if not -90 <= lean <= 120:
                    continue                     # 뒤집힌 자세 제외
                px = [d.geom_xpos[g][0] for g in pads]; pr = m.geom_size[pads[0]][0]
                sole = min(d.geom_xpos[g][2] for g in pads) - m.geom_size[pads[0]][1]
                cx = d.subtree_com[1][0]
                if not (min(px) - pr + margin <= cx <= max(px) + pr - margin):
                    continue
                hs = d.xpos[armR][2] - sole; side = "front" if lean >= 0 else "back"
                if side in best and hs >= best[side]["shoulder_h_mm"] / 1e3:
                    continue
                mujoco.mj_forward(m, d)
                dl = min(mujoco.mj_geomDistance(m, d, a, b, 0.1, ft) for a in legg for b in bodyg)
                if dl < clear:
                    continue
                best[side] = dict(shoulder_h_mm=hs * 1e3, K_deg=math.degrees(K), A_deg=math.degrees(A), H_deg=math.degrees(H),
                                  torso_lean_deg=lean, leg_body_mm=dl * 1e3, com_x_rel_mm=(cx - np.mean(px)) * 1e3)
    return best


if __name__ == "__main__":
    out = []
    for hm, km in ((-80, 100), (-115, 100), (-115, 115)):
        b = low_shoulder2(ARMS["팔꿈치 110+110"], km, hm)
        for side, v in b.items():
            out.append(dict(hip_min_deg=hm, knee_max_deg=km, side=side, **v))
            print(hm, km, side, {k: round(float(x), 1) for k, x in v.items()}, flush=True)
    jdump(out, os.path.join(RES, "static2.json"))
