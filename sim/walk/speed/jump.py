"""질문 1 · 2: 두 발이 모두 뜨는 동작을 서보가 낼 수 있는지 — 단순 수직 도약 시험(제어기 없이 키프레임 + 최대 힘).

1) 0–1 s: 서 있는 영점에서 웅크림 자세(고관절 피치 −a, 무릎 2a, 발목 피치 −a: 허벅지 = 정강이 75 mm 라 엉덩이가 발목 위로 곧게 내려감)로
   이득 2배 위치 제어로 내려가고 0.6 s 버팀.
2) 1.6 s: 무릎 목표를 0(다 폄)으로 바꾸고 다리 서보 위치 이득을 20배로 → 무릎은 정지토크에서 포화(서보가 낼 수 있는 최대 힘,
   토크-속도 직선은 모델의 감쇠 b = τs/ω0 가 만든다). 고관절 피치 · 발목 피치 목표는 지금 무릎 각의 절반(−q무릎/2)을 따라가게 해
   몸통이 곧게 위로 올라가게 하고, 발목은 펴는 동안 +p(발끝 내림)를 더한다. 무릎이 각 2a 를 돌아야 해서 고관절 · 발목보다 두 배 빨리 돈다
   → 이 동작의 병목은 무릎. (먼저 해 본 「세 관절 모두 최대 힘으로 폄」은 고관절이 먼저 펴져 몸통이 뒤로 젖혀지며 넘어짐.)
3) 잰 것: 발이 모두 바닥에서 떨어진 가장 긴 시간(체공), 무게중심 최고 높이 − 서 있는 높이, 최대 위로 속도, 관절별 최대 |ω|/ω0.
「한 발 뛰기 근사」: 다리 서보 정지토크 · 감쇠를 둘 다 절반(무부하 속도 그대로) — 한 다리가 몸무게 전부를 미는 것과 같은 부하 [근사].

usage: cd sim/walk/speed && ../../.venv/bin/python jump.py → results/jump.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import mujoco
from sp_common import CUR, hrA, Idx2, SEG
from kw_common import build_model, scale_kp, jdump
from walk_sweep import SERVOS, servo_cfg

LEGJ = ("hip_pitch", "knee", "ankle_pitch", "hip_roll", "ankle_roll", "hip_yaw")


def jump(cfg, a=0.5, p=0.3, half=False, T=3.0, trace=None):
    m, _ = build_model(**cfg)
    scale_kp(m, 2.0)
    ix = Idx2(m)
    if half:
        for i, nm in enumerate(ix.act):
            if nm[2:] in LEGJ:
                m.actuator_forcerange[i] *= 0.5; m.dof_damping[ix.vadr[ix.act_j[i]]] *= 0.5
                m.actuator_gainprm[i, 0] *= 0.5; m.actuator_biasprm[i, 1] *= 0.5
        ix = Idx2(m)                     # 절반으로 줄인 정지토크로 다시(사용률 분모)
    d = mujoco.MjData(m); d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
    z_stand = float(d.subtree_com[1][2])
    kp0 = m.actuator_gainprm[:, 0].copy()
    leg = [i for i, nm in enumerate(ix.act) if nm[2:] in LEGJ]
    kn = [ix.qadr[f"{s}_knee"] for s in "RL"]

    def tgt(s, ext):
        q = np.zeros(m.nu)
        qk = float(np.mean(d.qpos[kn]))           # 펼 때: 무릎은 최대 힘으로 0 을 향하고, 고관절 · 발목은 지금 무릎 각의 절반을 따라감
        for i, nm in enumerate(ix.act):
            j = nm[2:]
            if j == "hip_pitch": q[i] = -qk / 2 if ext else -a * s
            if j == "knee": q[i] = 0.0 if ext else 2 * a * s
            if j == "ankle_pitch": q[i] = (-qk / 2 + p * (1 - qk / (2 * a))) if ext else -a * s
        return q
    floor = ix.floor
    t_jump = 1.6; n = int(T / m.opt.timestep)
    v_to = x_to = tilt_to = None; air = []; com = []; vz = []; wmax = np.zeros(m.nu); tq = np.zeros(m.nu); took = None
    for k in range(n):
        t = k * m.opt.timestep
        if t < t_jump:
            d.ctrl[:] = tgt(min(1.0, t / 1.0), False)
        else:
            if k == int(t_jump / m.opt.timestep):
                m.actuator_gainprm[leg, 0] = kp0[leg] * 10; m.actuator_biasprm[leg, 1] = -kp0[leg] * 10
            d.ctrl[:] = tgt(1.0, True)
        mujoco.mj_step(m, d)
        if trace is not None and k % 10 == 0 and t > t_jump - 0.05:
            nc = sum(1 for c in d.contact[:d.ncon] if floor in (c.geom1, c.geom2))
            trace.append((round(t, 3), round(d.subtree_com[1][2] * 1e3, 1), round(d.subtree_com[1][0] * 1e3, 1),
                          round(math.degrees(math.asin(-d.xmat[1][6])), 1), nc, round(float(np.mean(d.qpos[kn])), 2)))
        if t >= t_jump:
            on = any(floor in (c.geom1, c.geom2) for c in d.contact[:d.ncon])
            on = on or d.xmat[1][8] < math.cos(math.radians(20))      # 몸통이 20° 넘게 기울면 체공으로 안 셈
            air.append(not on); com.append(d.subtree_com[1][2]); vz.append(float(d.cvel[1][5]) if False else None)
            w = np.array([d.qvel[ix.vadr[j]] for j in ix.act_j])
            if took is None:
                wmax = np.maximum(wmax, np.abs(w) / ix.w0)
                b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
                tq = np.maximum(tq, np.abs(d.actuator_force - b * w) / ix.stall)
                if not on:
                    took = t - t_jump
                    mujoco.mj_subtreeVel(m, d)
                    v_to = float(d.subtree_linvel[1][2]); x_to = float(d.subtree_com[1][0]); tilt_to = math.degrees(math.acos(min(1, d.xmat[1][8])))
    air = np.array(air); com = np.array(com)
    # 가장 긴 연속 체공
    best = cur = 0
    for x in air:
        cur = cur + 1 if x else 0; best = max(best, cur)
    zc = np.gradient(com, m.opt.timestep)
    R3 = d.xmat[1].reshape(3, 3)
    return dict(flight_s=best * m.opt.timestep, takeoff_s=took, v_takeoff=v_to, comx_takeoff_mm=x_to and x_to * 1e3, tilt_takeoff_deg=tilt_to, com_rise_mm=(com.max() - z_stand) * 1e3,
                com_rise_from_squat_mm=(com.max() - com[0]) * 1e3, vz_max=float(zc.max()), z_stand=z_stand,
                w_over_w0={nm[2:]: float(wmax[i]) for i, nm in enumerate(ix.act) if nm.startswith("R_")},
                torque_over_stall={nm[2:]: float(tq[i]) for i, nm in enumerate(ix.act) if nm.startswith("R_")},
                upright_end=bool(R3[2, 2] > math.cos(math.radians(35))))


def job(arg):
    name, cfg, a, p, half = arg
    try:
        return dict(name=name, a=a, p=p, half=half, **jump(cfg, a, p, half))
    except Exception as e:
        return dict(name=name, a=a, p=p, half=half, error=repr(e))


if __name__ == "__main__":
    from multiprocessing import Pool
    cfgs = {"지금 (늘림 40.5 mm)": dict(hip_roll=hrA(0.0405), servo_map=CUR), "지금 (늘림 0 mm)": dict(hip_roll=hrA(0.0), servo_map=CUR)}
    for n in ("고관절 피치 · 무릎 W210", "다리 XM 모두 W210", "피치 · 무릎 W210 + 발목 피치 W150", "발목 피치 XM430-W350",
              "피치 · 무릎 W210 + 발목 피치 XM430-W210"):
        c = servo_cfg(SERVOS[n]); cfgs[n] = c
    args = [(n, c, a, p, h) for n, c in cfgs.items() for a in (0.3, 0.5, 0.7) for p in (0.0, 0.2, 0.35) for h in (False, True)]
    with Pool(min(40, len(args))) as P:
        rows = P.map(job, args)
    jdump(rows, os.path.join(HERE, "results", "jump.json"))
    for r in rows:
        if "error" in r: print(r); continue
        print(f"{r['name'][:28]:28s} a={r['a']} p={r['p']} half={r['half']!s:5} flight={r['flight_s']*1e3:5.1f}ms rise={r['com_rise_mm']:6.1f} "
              f"vto={r['v_takeoff']} took={r['takeoff_s']} tilt={r['tilt_takeoff_deg']} wknee={r['w_over_w0']['knee']:.2f} whip={r['w_over_w0']['hip_pitch']:.2f} "
              f"wank={r['w_over_w0']['ankle_pitch']:.2f} up={r['upright_end']}")


def arm_sensitivity():
    """회전자 관성(armature, 추정값) 절반일 때 같은 도약 — 결론이 관성 추정에 얼마나 기대는지"""
    from multiprocessing import Pool
    cf = {"지금 (늘림 40.5 mm)": dict(hip_roll=hrA(0.0405), servo_map=CUR),
          "피치 · 무릎 W210 + 발목 피치 W150": servo_cfg(SERVOS["피치 · 무릎 W210 + 발목 피치 W150"])}
    args = [(n + " · 관성 ×0.5", dict(c, armature=0.5), a, 0.0, h) for n, c in cf.items() for a in (0.5, 0.7) for h in (False, True)]
    with Pool(len(args)) as P:
        rows = P.map(job, args)
    jdump(rows, os.path.join(HERE, "results", "jump_armature.json"))
    for r in rows:
        print(r["name"], r["a"], r["half"], f"flight={r['flight_s']*1e3:.0f}ms rise={r['com_rise_mm']:.1f} up={r['upright_end']} wknee={r['w_over_w0']['knee']:.2f}")
