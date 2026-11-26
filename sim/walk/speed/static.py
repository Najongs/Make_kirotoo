"""정적 관절 토크(준정적, 동역학 없음): 다리 길이 변형별
 (1) 한 발 정지: 웅크림 c(15 · 25 · 35 mm) 자세에서 몸무게 전부를 한 발이 받칠 때 무릎 토크.
 (2) 깊은 쪼그림: 무릎 K 도, 두 발 평평, 발목 피치 −50°(지금 범위 끝, 발등 굽힘 최대)에서 무게중심이 발 가운데 위에 오도록
     고관절 피치 H 를 찾음 → 무릎 토크(두 발이 반씩), 필요한 고관절 굽힘, 엉덩이 높이. 「쪼그려 앉은 자세에서 일어날 때」 가장 힘든 곳.
 자세: 시상면(앞뒤)만, 좌우 대칭. 토크 = g(q) − Jᵀ F (발바닥 압력중심에 몸무게), WholeBodyIK.static_torque 그대로.
usage: cd sim/walk/speed && ../../.venv/bin/python static.py → results/static.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
from sp_common import CUR, hrA, SEG, build_model2
from kw_ctrl import WholeBodyIK
from kw_common import jdump, Idx

G = 9.81


def model(L, ext):
    SEG.update(thigh=ext, shank=ext, mass=True, servo=CUR)
    m, _ = build_model2(hip_roll=hrA(L), servo_map=CUR)
    SEG.update(thigh=0.0, shank=0.0, mass=False)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return m


def pose(m, ix, ik, H, K, A, one_leg=False):
    q = m.qpos0.copy(); q[3:7] = [1, 0, 0, 0]; q[:3] = 0
    for s in "RL":
        q[ix.qadr[f"{s}_hip_pitch"]] = H; q[ix.qadr[f"{s}_knee"]] = K; q[ix.qadr[f"{s}_ankle_pitch"]] = A
    d = ik.d; d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    # 발이 평평하도록 몸 전체를 y 축으로 돌림
    Rf = d.xmat[ix.foot["R"]].reshape(3, 3); ang = math.atan2(Rf[0, 2], Rf[2, 2])
    qq = np.zeros(4); mujoco.mju_axisAngle2Quat(qq, np.array([0, 1.0, 0]), -ang); q[3:7] = qq
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    foot = d.xpos[ix.foot["R"]].copy(); com = d.subtree_com[1].copy()
    return q, com, foot, d.xpos[ix.body("R_thigh")][2] - foot[2]


def knee_torque(m, ix, ik, q, com, two):
    mg = ix.mass * G; d = ik.d; d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    loads = []
    for s in ("RL" if two else "R"):
        p = d.xpos[ix.foot[s]].copy(); p[0] = com[0]; p[2] -= 0.035
        loads.append((ix.foot[s], p, np.array([0, 0, mg / (2 if two else 1)])))
    tau = ik.static_torque(q, loads)
    return {j: float(abs(tau[ix.vadr[f"R_{j}"]])) for j in ("hip_pitch", "knee", "ankle_pitch")}


def solve_H(m, ix, ik, K, A, target_dx=0.0):
    """무게중심 x − 발 가운데 x = target_dx 가 되는 H (이분법, −3 ~ 0.6 rad)"""
    pad_c = np.mean([m.geom_pos[g][:2] for g in ix.pads["R"]], axis=0)[0]
    f = lambda H: (lambda r: r[1][0] - (r[2][0] + pad_c) - target_dx)(pose(m, ix, ik, H, K, A))
    lo, hi = -3.0, 0.6
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    out = []
    for L in (0.0, 0.02, 0.0405):
        for ext in (0.0, 0.01, 0.02, 0.03):
            m = model(L, ext); ix = Idx(m); ik = WholeBodyIK(m); lk = 0.075 + ext
            row = dict(L=L, ext=ext, mass=ix.mass, one_leg={}, squat={})
            for c in (0.015, 0.025, 0.035):
                a = math.acos((2 * lk - c) / (2 * lk))       # 엉덩이가 c 만큼 내려가는 대칭 자세(고관절 −a, 무릎 2a, 발목 −a)
                H = solve_H(m, ix, ik, 2 * a, -a) or -a
                q, com, foot, hh = pose(m, ix, ik, H, 2 * a, -a)
                t = knee_torque(m, ix, ik, q, com, two=False)
                row["one_leg"][c] = dict(knee_deg=math.degrees(2 * a), hip_deg=math.degrees(H), **{k + "_Nm": v for k, v in t.items()})
            for Kd in (60, 80, 100, 110, 120, 130, 140, 150):
                K = math.radians(Kd); A = -0.873
                H = solve_H(m, ix, ik, K, A)
                if H is None:
                    row["squat"][Kd] = None; continue
                q, com, foot, hh = pose(m, ix, ik, H, K, A)
                t = knee_torque(m, ix, ik, q, com, two=True)
                row["squat"][Kd] = dict(hip_deg=math.degrees(H), hip_height_mm=hh * 1e3, com_height_mm=(com[2] - foot[2] + 0.035) * 1e3,
                                        **{k + "_Nm": v for k, v in t.items()})
            out.append(row)
            print(f"L={L*1e3:4.1f} ext={ext*1e3:2.0f} m={ix.mass:.3f} one-leg knee Nm " +
                  " ".join(f"c{int(c*1e3)}:{v['knee_Nm']:.2f}" for c, v in row["one_leg"].items()) + " | squat knee Nm (hip deg) " +
                  " ".join(f"{k}:{v['knee_Nm']:.2f}({v['hip_deg']:.0f},{v['ankle_pitch_Nm']:.2f})" if v else f"{k}:-" for k, v in row["squat"].items()))
    jdump(out, os.path.join(HERE, "results", "static.json"))
