"""제자리 돌기 25° · 30°/걸음이 왜 넘어지나: 두 다리끼리 접촉(col_leg bit 4 쌍), 넘어지기 전 1 초 포화, 고관절 요 힘 튐 시각.
python turn_fall.py → results/turn_fall.json"""
import math, os
from collections import deque
import numpy as np
import mujoco
from kw_common import build_model, scale_kp, Idx, sense, RES, jdump
from kw_ctrl import Controller, Footsteps, make_steps
from run_tests import BASE


def run_turn(deg):
    c = dict(BASE)
    m, _ = build_model(); scale_kp(m, c["kp_scale"])
    mp, _ = build_model(); mp.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m); d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
    ix = Idx(m); kp = m.actuator_gainprm[:, 0].copy()
    dtc = 1.0 / c["ctrl_hz"]; nsub = int(round(dtc / m.opt.timestep))
    plan = Footsteps(make_steps(c["n_steps"], 0.0, yaw_step=math.radians(deg)), 1.5, c["Ts"], c["Td"], 0.8, 0.8, (-0.007, 0.0))
    ctl = Controller(mp, kp, dtc, plan, crouch=c["crouch"], lift=c["lift"], k_dcm=c["k_dcm"])
    nd = int(round(c["delay_ms"] / 1000 / dtc)); buf = deque([d.ctrl.copy()] * (nd + 1), maxlen=nd + 1)
    body = lambda g: m.body(m.geom_bodyid[g]).name
    contacts, sat, yawf, t_all = [], [], [], []
    fall = None
    a_yaw = [ix.aid["R_hip_yaw"], ix.aid["L_hip_yaw"]]
    T_end = 1.5 + 0.8 + (plan.n - 2) * (c["Ts"] + c["Td"]) + 1.3
    for k in range(int(T_end / m.opt.timestep)):
        t = k * m.opt.timestep
        if k % nsub == 0:
            buf.append(ctl.step(t, sense(m, d, ix))); d.ctrl[:] = buf[0]
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            cn = d.contact[i]; b1, b2 = body(cn.geom1), body(cn.geom2)
            if b1[:2] in ("R_", "L_") and b2[:2] in ("R_", "L_") and b1[0] != b2[0]:
                contacts.append((round(t, 3), m.geom(cn.geom1).name, m.geom(cn.geom2).name))
        if k % 5 == 0:
            t_all.append(t); sat.append(np.abs(d.actuator_force) >= 0.999 * m.actuator_forcerange[:, 1])
            yawf.append(np.abs(d.actuator_force[a_yaw]) / 1.75)
        if fall is None and d.xmat[ix.torso][8] < math.cos(math.radians(35)):
            fall = t
        if fall is not None and t > fall + 0.3:
            break
    t_all = np.array(t_all); sat = np.array(sat); yawf = np.array(yawf).max(1)
    tf = fall or t_all[-1]
    sel = (t_all > tf - 1.0) & (t_all <= tf)
    first_c = contacts[0][0] if contacts else None
    spikes = t_all[yawf > 0.6]
    return {"deg": deg, "fall_t": fall, "leg_leg_contacts": len(contacts), "first_leg_leg_contact": contacts[:3],
            "leg_leg_contact_times": sorted({c_[0] for c_ in contacts})[:20],
            "sat_last_1s": {ix.act[i]: float(sat[sel, i].mean()) for i in range(m.nu) if sat[sel, i].mean() > 0.01},
            "hip_yaw_force_over_0.6_times": [round(float(x), 3) for x in spikes[:20]],
            "hip_yaw_spike_with_contact": bool(first_c is not None and len(spikes) and abs(spikes[0] - first_c) < 0.05)}


if __name__ == "__main__":
    out = [run_turn(dg) for dg in (20, 25, 30)]
    for o in out:
        print(o)
    jdump(out, os.path.join(RES, "turn_fall.json"))
