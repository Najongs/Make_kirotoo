"""한 번의 시험(에피소드)을 돌린다: 물리 1 kHz, 제어 100/200 Hz(영차 유지), 명령 지연(링 버퍼), 밀기, 머리 끄덕임, 영상.
제어기는 sense() 만 받고, 평가는 정답 상태로 따로 잰다."""
import math, os, time
from collections import deque
import numpy as np
import mujoco
from kw_common import build_model, scale_kp, Idx, sense, true_cop_margin, utilisation, RES
from kw_ctrl import Controller, Footsteps, make_steps, rpy_of

DEFAULT = dict(
    kp_scale=2.0, ctrl_hz=200, delay_ms=0.0, friction=None, head_mass=1.0, pad_dy_out=0.0, pad_dx=0.0, armature=1.0,
    hip_roll=False, extra_mass=(), k_lat=0.0, lat_db=0.0, lat_in=True, record_fall=False,
    # 걸음
    n_steps=20, stride=0.0, yaw_step=0.0, Ts=0.36, Td=0.14, Tinit=0.8, Tend=0.8, t_walk=1.5, crouch=0.015, lift=0.015,
    zmp_off=(-0.007, 0.0), ramp=2,
    # 제어
    k_dcm=2.0, k_step=0.0, ff=True, posture=(0.0, 0.0, 0.0, 0.0), vel_fc=12.0, pitch=0.0, dcm_to_com=0.0, land_press=0.0, dyn_ff=True, lead_s=0.0,
    # 머리 끄덕임: (주파수 Hz, 시작 s) — 머리 −10~+20° 사인
    nod=None,
    # 밀기: (시작 s, 지속 s, 힘 N, 방향 'y'/'x')
    push=None,
    T_end=None, video=None, eye_video=None, eye_fps=30, eye_max_s=10.0, record=True,
)


def run(**kw):
    cfg = dict(DEFAULT); cfg.update(kw)
    m, spec = build_model(friction=cfg["friction"], head_mass=cfg["head_mass"], pad_dy_out=cfg["pad_dy_out"],
                          pad_dx=cfg["pad_dx"], armature=cfg["armature"], camera=bool(cfg["eye_video"]),
                          hip_roll=cfg["hip_roll"], extra_mass=cfg["extra_mass"])
    scale_kp(m, cfg["kp_scale"])
    # 계획용 모델: 같은 설계(머리 질량 · 패드 · 고관절 롤은 설계 변경이라 제어기도 안다), 접촉 · 등식 없음
    mp, _ = build_model(head_mass=cfg["head_mass"], pad_dy_out=cfg["pad_dy_out"], pad_dx=cfg["pad_dx"],
                        hip_roll=cfg["hip_roll"], extra_mass=cfg["extra_mass"])
    mp.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    ix = Idx(m)
    kp = m.actuator_gainprm[:, 0].copy()
    dtc = 1.0 / cfg["ctrl_hz"]; nsub = int(round(dtc / m.opt.timestep))
    steps = make_steps(cfg["n_steps"], cfg["stride"], yaw_step=cfg["yaw_step"], ramp=cfg["ramp"])
    plan = Footsteps(steps, cfg["t_walk"], cfg["Ts"], cfg["Td"], cfg["Tinit"], cfg["Tend"], cfg["zmp_off"])
    T_plan = cfg["t_walk"] + cfg["Tinit"] + (plan.n - 2) * (cfg["Ts"] + cfg["Td"]) + cfg["Tend"] + 0.5
    T_end = cfg["T_end"] or T_plan
    head_fn = None
    if cfg["nod"]:
        f, t0 = cfg["nod"]
        lo, hi = math.radians(-10), math.radians(20)
        # t0 전엔 0°. t0 부터 −10~+20° 사인(평균 5°, 진폭 15°)을 1 초에 걸쳐 키운다(계단 방지)
        head_fn = lambda t: (min(1.0, (t - t0)) * (0.5 * (lo + hi) - 0.5 * (hi - lo) * math.cos(2 * math.pi * f * (t - t0)))) if t > t0 else 0.0
    ctl = Controller(mp, kp, dtc, plan, crouch=cfg["crouch"], lift=cfg["lift"], k_dcm=cfg["k_dcm"], k_step=cfg["k_step"],
                     ff=cfg["ff"], posture=cfg["posture"], head_fn=head_fn, vel_fc=cfg["vel_fc"], pitch=cfg["pitch"],
                     dcm_to_com=cfg["dcm_to_com"], land_press=cfg["land_press"],
                     dyn_ff=cfg["dyn_ff"], lead_s=cfg["lead_s"], k_lat=cfg["k_lat"], lat_db=cfg["lat_db"], lat_in=cfg["lat_in"])
    nd = int(round(cfg["delay_ms"] / 1000 / dtc))
    buf = deque([d.ctrl.copy()] * (nd + 1), maxlen=nd + 1)
    # 기록
    rec = {k: [] for k in ("t", "com", "c_est", "c_ref", "p_ref", "p_d", "cop", "margin", "rpy", "fs", "u", "sat", "tau",
                           "head_w", "head_rpy", "support", "crank_tau", "phase", "head_deg", "qerr", "ik_roll", "ik_com", "meas_roll", "ctrl", "qmeas", "w_act")}
    fall_t = None; ren = None; frames = []; eye_frames = []
    if cfg["video"] or cfg["eye_video"]:
        ren = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    every = int(round(1.0 / (cfg["eye_fps"] * m.opt.timestep)))
    n = int(T_end / m.opt.timestep)
    push = cfg["push"]
    tb = ix.torso
    torso_sub = None
    wall0 = time.time()
    a_crank = ix.aid["nod_crank"]
    for k in range(n):
        t = k * m.opt.timestep
        if k % nsub == 0:
            sen = sense(m, d, ix)
            u_cmd = ctl.step(t, sen)
            buf.append(u_cmd)
            d.ctrl[:] = buf[0]
            if cfg["record"] and fall_t is None:
                lg = ctl.log
                rec["t"].append(t); rec["c_est"].append(lg["c"]); rec["c_ref"].append(lg["c_ref"]); rec["p_ref"].append(lg["p_ref"])
                rec["p_d"].append(lg["p_d"]); rec["support"].append(lg["support"]); rec["phase"].append(lg["phase"])
                rec["ik_roll"].append(lg["ik_roll"]); rec["ik_com"].append(lg["ik_com"]); rec["meas_roll"].append(lg["meas_roll"])
                rec["ctrl"].append(u_cmd.copy()); rec["qmeas"].append(sen["q"].copy())
        d.xfrc_applied[tb] = 0
        if push and push[0] <= t < push[0] + push[1]:
            F = np.zeros(6); F[{"x": 0, "y": 1}[push[3]]] = push[2]
            # 몸 전체 무게중심 높이에 수평으로(몸통 body 에 가하되 힘 작용점을 무게중심으로: 모멘트 보정)
            r = d.subtree_com[tb] - d.xipos[tb]
            F[3:] = np.cross(r, F[:3])
            d.xfrc_applied[tb] = F
        mujoco.mj_step(m, d)
        R3 = d.xmat[tb].reshape(3, 3)
        tilt = math.degrees(math.acos(np.clip(R3[2, 2], -1, 1)))
        if fall_t is None and (tilt > 35 or d.subtree_com[tb][2] < 0.13):
            fall_t = t
        if cfg["record"] and (fall_t is None or cfg["record_fall"]) and k % 5 == 0:
            cop, mg, Fz = true_cop_margin(m, d, ix)
            fs, u, sat = utilisation(m, d, ix)
            rec["com"].append(d.subtree_com[tb].copy()); rec["cop"].append(cop if cop is not None else np.full(2, np.nan))
            rec["margin"].append(mg if mg is not None else np.nan)
            y, p, r_ = rpy_of(R3); rec["rpy"].append((r_, p, y))
            rec["fs"].append(fs); rec["u"].append(u); rec["sat"].append(sat); rec["tau"].append(d.actuator_force.copy())
            rec["w_act"].append(np.array([d.qvel[ix.vadr[j]] for j in ix.act_j]))
            # 카메라(머리) 각속도: 머리 body 각속도 world
            w = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ix.head, w, 0)
            rec["head_w"].append(w[:3].copy())
            Rh = d.xmat[ix.head].reshape(3, 3); rec["head_rpy"].append(rpy_of(Rh))
            rec["crank_tau"].append(d.actuator_force[a_crank])
            rec["head_deg"].append(math.degrees(d.qpos[ix.qadr["head_nod"]]))
        if ren is not None and k % every == 0 and t <= cfg["eye_max_s"] + cfg.get("eye_t0", 0) and t >= cfg.get("eye_t0", 0):
            if cfg["video"]:
                cam.lookat[:] = [d.subtree_com[tb][0], d.subtree_com[tb][1], 0.2]; cam.distance = 0.9
                cam.azimuth = 215; cam.elevation = -12
                ren.update_scene(d, cam); frames.append(ren.render().copy())
            if cfg["eye_video"]:
                ren.update_scene(d, camera="eye"); eye_frames.append(ren.render().copy())
        if fall_t is not None and t > fall_t + cfg.get("after_fall_s", 0.5):
            break
    import imageio.v2 as iio
    if cfg["video"] and frames:
        iio.mimsave(cfg["video"], frames, fps=cfg["eye_fps"], codec="libx264", quality=7, macro_block_size=8)
    if cfg["eye_video"] and eye_frames:
        iio.mimsave(cfg["eye_video"], eye_frames, fps=cfg["eye_fps"], codec="libx264", quality=7, macro_block_size=8)
    R = {k: np.array(v) for k, v in rec.items() if k not in ("support", "phase")}
    R["support"] = rec["support"]; R["phase"] = rec["phase"]
    out = summarize(m, ix, cfg, plan, R, fall_t, T_end)
    out["wall_s"] = time.time() - wall0
    return out, R, m


def summarize(m, ix, cfg, plan, R, fall_t, T_end):
    names = ix.act
    res = {"fallen": fall_t is not None, "fall_t": fall_t, "T_end": T_end}
    t_walk = cfg["t_walk"] + cfg["Tinit"]; T = cfg["Ts"] + cfg["Td"]
    tf = fall_t if fall_t is not None else T_end
    res["steps_done"] = int(max(0, min(plan.n - 2, (tf - t_walk) // T)))
    com = R["com"]
    if len(com):
        res["com_dx_m"] = float(com[-1, 0] - com[0, 0]); res["com_dy_m"] = float(com[-1, 1] - com[0, 1])
        nominal = sum(s["pos"][0] for s in plan.steps[-1:]) if False else plan.steps[-1]["pos"][0]
        res["nominal_dx_m"] = float(nominal)
        # 정속 구간 평균 속도: 걸음 시작 + 3걸음 ~ 끝 2걸음 전
        tt = np.arange(len(com)) * 5e-3
        a, b = t_walk + 3 * T, min(tf, t_walk + (plan.n - 4) * T)
        sel = (tt >= a) & (tt <= b)
        res["speed_mps"] = float((com[sel][-1, 0] - com[sel][0, 0]) / (tt[sel][-1] - tt[sel][0])) if sel.sum() > 10 else None
        rpy = np.degrees(R["rpy"])
        res["torso_roll_deg"] = [float(rpy[:, 0].min()), float(rpy[:, 0].max())]
        res["torso_pitch_deg"] = [float(rpy[:, 1].min()), float(rpy[:, 1].max())]
        res["torso_yaw_deg"] = [float(rpy[:, 2].min()), float(rpy[:, 2].max()), float(rpy[-1, 2])]
        mg = R["margin"]; walk = tt >= t_walk
        st = tt >= 1.0          # 시작 1 초(영점 → 웅크림 섞기) 제외
        mgw = mg[walk]
        res["cop_margin_mm"] = {"min": float(np.nanmin(mgw) * 1e3) if walk.any() else None,
                                "p5": float(np.nanpercentile(mgw, 5) * 1e3) if walk.any() else None,
                                "outside_frac": float(np.mean(mgw < 0)) if walk.any() else None}
        fs, u, sat = R["fs"], R["u"], R["sat"]
        res["force_over_stall_max"] = {nm: float(fs[st, i].max()) for i, nm in enumerate(names)}
        res["force_over_stall_p99"] = {nm: float(np.percentile(fs[st, i], 99)) for i, nm in enumerate(names)}
        res["u_max"] = {nm: float(u[st, i].max()) for i, nm in enumerate(names)}
        res["u_p99"] = {nm: float(np.percentile(u[st, i], 99)) for i, nm in enumerate(names)}
        res["sat_frac"] = {nm: float(sat[st, i].mean()) for i, nm in enumerate(names)}
        res["u_over1_frac"] = {nm: float((u[st, i] > 1).mean()) for i, nm in enumerate(names)}
        hw = np.degrees(R["head_w"])
        res["head_w_rms_dps"] = float(np.sqrt((hw[walk] ** 2).sum(1).mean())) if walk.any() else None
        res["head_w_max_dps"] = float(np.sqrt((hw[walk] ** 2).sum(1)).max()) if walk.any() else None
        res["crank_tau_max"] = float(np.abs(R["crank_tau"]).max())
        res["crank_tau_rms"] = float(np.sqrt((R["crank_tau"][walk] ** 2).mean())) if walk.any() else None
        res["head_deg_range"] = [float(R["head_deg"].min()), float(R["head_deg"].max())]
    return res


def brief(o):
    s = f"fall={o['fallen']}({o['fall_t']}) steps={o['steps_done']} dx={o.get('com_dx_m', 0):.3f}/{o.get('nominal_dx_m', 0):.3f}"
    if o.get("speed_mps") is not None:
        s += f" v={o['speed_mps']:.3f}"
    if "torso_roll_deg" in o:
        s += f" roll={o['torso_roll_deg'][0]:.1f}..{o['torso_roll_deg'][1]:.1f} pitch={o['torso_pitch_deg'][0]:.1f}..{o['torso_pitch_deg'][1]:.1f}"
        s += f" yawEnd={o['torso_yaw_deg'][2]:.1f} margin(min,p5,out)={o['cop_margin_mm']['min']:.1f},{o['cop_margin_mm']['p5']:.1f},{o['cop_margin_mm']['outside_frac']:.2f}" if o['cop_margin_mm']['min'] is not None else ""
        um = o["u_max"]; top = sorted(um.items(), key=lambda kv: -kv[1])[:3]
        s += " umax " + ",".join(f"{k}:{v:.2f}" for k, v in top)
    s += f" wall={o.get('wall_s', 0):.0f}s"
    return s
