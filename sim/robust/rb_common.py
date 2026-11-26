"""강건성 검사 공용: sim/walk 제어기를 그대로 쓰고, 「실물이 설계와 다를 때」를 물리 모델(plant)에만 넣는다.

제어기(계획용 모델)는 설계값만 안다. 아래 흐트러짐은 물리 모델에만 들어간다 — 제어기는 모른다.
  - 바닥 마찰(friction): 모든 충돌 geom 미끄럼 마찰(build_model 과 같은 방법). 기준 0.8 = kitoru_v18.xml 기본값.
  - 기어 백래시(backlash_deg): 다리 서보 관절마다 같은 축 · 같은 자리에 「놀이 관절」을 하나 더 넣고(범위 ±b/2, 질량 없음),
      서보 관절(모터 쪽)에는 회전자 관성 · 감쇠 · 서보 힘이 그대로 남는다. 바깥 짐은 놀이 관절 범위 안에서는 모터에 닿지 않는다.
      X430 엔코더(AS5045/AS5601 절대 12 bit, 출력 한 바퀴 360°)는 출력축을 읽는다고 봄 [추정: 출력 한 바퀴를 절대각으로 주므로]
      → 서보 P 제어는 출력각(모터 각 + 놀이 각)으로 닫힌다: 물리 1 kHz 마다 ctrl = 명령 − 놀이 각. 제어기 센서도 출력각을 읽는다.
      놀이 끝 부딪힘은 관절 한계 solref 0.002 s(물리 시간 간격의 2 배, MuJoCo 가 허용하는 가장 단단한 값) · solimp 0.99–0.999.  e-manual XM430-W350 백래시 15′(0.25°).
  - 내부 마찰(frictionloss_frac): 다리 서보 관절 frictionloss = 정지토크 × 비율 [추정: e-manual 에 값 없음].
  - 전압(volt): 정지토크 · 무부하속도를 e-manual 표(9 · 11.1 · 12 V 등) 직선 보간으로, 위치 이득 kp 는 V/11.1 배
      (P 이득 레지스터가 PWM 듀티를 정하므로 토크/오차가 전압에 비례 [추정]). 감쇠 = 정지토크/무부하속도로 다시 계산.
  - 제어 지연(delay_ms): kw_sim 과 같은 링 버퍼(제어 주기 5 ms 단위).
  - 질량(mass_f): 모든 body 질량 · 관성 배수(무게중심 그대로). 제어기는 설계 질량으로 계획한다.
  - 무게중심 앞뒤(com_dx_m): 몸 전체 무게중심이 dx 만큼 옮겨지도록 몸통 body 무게중심을 dx·M/m_몸통 옮김.
기준 모델: v18 MJCF + 고관절 롤 안 A(직렬, XM430, hiproll/hr_cfg.py) + v20 서보 배치(battery6000/sim_runs.py V20).
  새 v20 형상 모델이 아직 없어 이 조합은 근사다(형상 · 질량은 v18, 롤 서보 질량 · 다리 +40.5 mm 만 반영).
배낭(backpack=True): battery6000/sim_runs.py pack(BACK, 0.250) — LB-012 · Jetson 배터리 빼고 강압 모듈 15 g, 250 g 팩 (−117, 0, 280) mm.
  배낭은 설계 결정이라 제어기 모델에도 넣는다."""
import sys, os, math, time
from collections import deque
HERE = os.path.dirname(os.path.abspath(__file__))
WALK = os.path.join(os.path.dirname(HERE), "walk")
for p in (WALK, os.path.join(WALK, "hiproll"), os.path.join(WALK, "battery6000")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import mujoco
from kw_common import build_model, scale_kp, Idx, sense, true_cop_margin, utilisation, jdump
from kw_ctrl import Controller, Footsteps, make_steps, rpy_of
from kw_sim import summarize, DEFAULT
from run_tests import BASE
from hr_cfg import hr
from sim_runs import V20, BACK, pack

RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
LEG_KEYS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
STRIDE = 0.05                      # 0.05 m / (0.45 + 0.20) s = 0.077 m/s (현황판 · 보고서의 「0.076 m/s」)
ROB0 = dict(BASE, hip_roll=hr("A"), servo_map=V20, k_lat=0.0)    # k_lat 0: hiproll compare.json 안 A 0.05 에서 여유가 더 큼
# e-manual 표 (전압 V, 정지토크 N·m, 무부하 rpm) [e-manual]
EMAN_V = {"XM430-W350": [(11.1, 3.8, 43), (12.0, 4.1, 46), (14.8, 4.8, 57)],
          "XC430-W240": [(9.0, 1.4, 52), (11.1, 1.7, 65), (12.0, 1.9, 70)],
          "XL430-W250": [(9.0, 1.0, 47), (11.1, 1.4, 57), (12.0, 1.5, 61)]}
TYPE_OF_STALL = {3.8: "XM430-W350", 1.75: "XC430-W240", 1.4: "XL430-W250"}


def volt_ratio(servo, V):
    """11.1 V 대비 (정지토크 비, 무부하속도 비): 표 직선 보간, 표 밖은 가까운 두 점으로 바깥 늘림"""
    t = EMAN_V[servo]; vs = np.array([r[0] for r in t]); ts = np.array([r[1] for r in t]); ws = np.array([r[2] for r in t])
    def lin(y):
        i = 0 if V <= vs[1] else len(vs) - 2
        if vs[0] <= V <= vs[-1]:
            return float(np.interp(V, vs, y))
        return float(y[i] + (y[i + 1] - y[i]) * (V - vs[i]) / (vs[i + 1] - vs[i]))
    t0 = float(np.interp(11.1, vs, ts)); w0 = float(np.interp(11.1, vs, ws))
    return lin(ts) / t0, lin(ws) / w0


class Idx2(Idx):
    """무부하속도를 관절 감쇠에서 다시 계산(전압을 바꾸면 forcerange 로 서보를 못 알아봄)"""
    def __init__(self, m):
        super().__init__(m)
        b = np.array([m.dof_damping[self.vadr[j]] for j in self.act_j])
        self.w0 = np.where(b > 1e-6, self.stall / np.maximum(b, 1e-9), 5.0)


def is_leg_act(name):
    return name[:2] in ("R_", "L_") and name[2:] in LEG_KEYS


def plant_mods(backlash_deg=0.0, frictionloss_frac=0.0, mass_f=1.0, com_dx_m=0.0, bl_solref=(0.002, 1.0), bl_solimp=(0.99, 0.999, 0.0005)):
    """물리 모델 MjSpec 을 컴파일 직전에 고치는 함수(build_model 의 extra_bodies 자리)"""
    def f(spec):
        stall = {a.target: float(a.forcerange[1]) for a in spec.actuators if is_leg_act(a.name)}
        if frictionloss_frac:
            for jn, ts in stall.items():
                spec.joint(jn).frictionloss = frictionloss_frac * ts
        if backlash_deg:
            h = math.radians(backlash_deg) / 2
            bodies = {}
            for jn in stall:
                bodies.setdefault(spec.joint(jn).parent.name, None)
            for bn in bodies:
                b = spec.body(bn)
                js = [j for j in b.joints]
                saved = []
                for j in js:
                    saved.append(dict(name=j.name, type=j.type, axis=np.array(j.axis), pos=np.array(j.pos), range=np.array(j.range),
                                      limited=j.limited, damping=np.array(j.damping), armature=j.armature, frictionloss=j.frictionloss,
                                      ref=j.ref, stiffness=np.array(j.stiffness), springref=j.springref,
                                      solref_limit=np.array(j.solref_limit), solimp_limit=np.array(j.solimp_limit), margin=j.margin))
                for j in js:
                    spec.delete(j)
                for s in saved:          # 원래 순서대로 다시 넣고, 서보 관절 바로 뒤에 놀이 관절
                    j = b.add_joint()
                    for k, v in s.items():
                        setattr(j, k, v)
                    if s["name"] in stall:
                        jb = b.add_joint(); jb.name = "bl_" + s["name"]; jb.type = mujoco.mjtJoint.mjJNT_HINGE
                        jb.axis = s["axis"]; jb.pos = s["pos"]; jb.range = [-h, h]; jb.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                        jb.damping = [1e-4, 0, 0]; jb.armature = 1e-6; jb.solref_limit = list(bl_solref); jb.solimp_limit = list(bl_solimp) + [0.5, 2.0]
        if mass_f != 1.0:
            for b in spec.bodies:
                if b.mass > 0:
                    b.mass = b.mass * mass_f; b.fullinertia = np.asarray(b.fullinertia) * mass_f
        if com_dx_m:
            M = sum(b.mass for b in spec.bodies if b.name != "world")
            t = spec.body("torso"); p = np.array(t.ipos, float); p[0] += com_dx_m * M / t.mass; t.ipos = p
    return f


def run_rb(**kw):
    """kw_sim.run 과 같은 흐름(밀기 · 끄덕임 · 영상 뺌) + 물리 모델 흐트러짐. 추가 키:
    backpack, backlash_deg, frictionloss_frac, volt, mass_f, com_dx_m, com_known(무게중심 어긋남을 제어기도 앎)."""
    cfg = dict(DEFAULT); cfg.update(ROB0); cfg.update(kw)
    em = tuple(cfg["extra_mass"]) + (pack(BACK, 0.250) if cfg.get("backpack") else ())
    pm = plant_mods(cfg.get("backlash_deg", 0.0), cfg.get("frictionloss_frac", 0.0), cfg.get("mass_f", 1.0), cfg.get("com_dx_m", 0.0),
                    cfg.get("bl_solref", (0.002, 1.0)), cfg.get("bl_solimp", (0.99, 0.999, 0.0005)))
    m, spec = build_model(friction=cfg["friction"], head_mass=cfg["head_mass"], pad_dy_out=cfg["pad_dy_out"], pad_dx=cfg["pad_dx"],
                          armature=cfg["armature"], hip_roll=cfg["hip_roll"], extra_mass=em, servo_map=cfg["servo_map"],
                          mass_scale=cfg["mass_scale"], extra_bodies=pm)
    scale_kp(m, cfg["kp_scale"])
    mp, _ = build_model(head_mass=cfg["head_mass"], pad_dy_out=cfg["pad_dy_out"], pad_dx=cfg["pad_dx"],
                        hip_roll=cfg["hip_roll"], extra_mass=em, servo_map=cfg["servo_map"], mass_scale=cfg["mass_scale"])
    mp.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if cfg.get("com_known") and cfg.get("com_dx_m"):
        # 실물 무게중심을 재서 제어기 모델에 넣은 경우: 계획용 모델의 몸통 무게중심도 같은 만큼 옮김
        tid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, "torso")
        mp.body_ipos[tid, 0] += cfg["com_dx_m"] * float(mp.body_subtreemass[tid]) / float(mp.body_mass[tid])
    kp = m.actuator_gainprm[:, 0].copy()          # 제어기는 설계 전압의 kp 를 안다(앞먹임 계산용)
    V = cfg.get("volt", 11.1)
    if V != 11.1:
        for i in range(m.nu):
            st = TYPE_OF_STALL.get(round(float(m.actuator_forcerange[i, 1]), 2))
            if st is None:
                continue
            rt, rw = volt_ratio(st, V); rk = V / 11.1
            m.actuator_forcerange[i] *= rt; m.actuator_gainprm[i, 0] *= rk; m.actuator_biasprm[i, 1] *= rk
            dof = m.jnt_dofadr[m.actuator_trnid[i][0]]; m.dof_damping[dof] *= rt / rw
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    ix = Idx2(m)
    dtc = 1.0 / cfg["ctrl_hz"]; nsub = int(round(dtc / m.opt.timestep))
    steps = make_steps(cfg["n_steps"], cfg["stride"], yaw_step=cfg["yaw_step"], ramp=cfg["ramp"])
    plan = Footsteps(steps, cfg["t_walk"], cfg["Ts"], cfg["Td"], cfg["Tinit"], cfg["Tend"], cfg["zmp_off"])
    T_plan = cfg["t_walk"] + cfg["Tinit"] + (plan.n - 2) * (cfg["Ts"] + cfg["Td"]) + cfg["Tend"] + 0.5
    T_end = cfg["T_end"] or T_plan
    ctl = Controller(mp, kp, dtc, plan, crouch=cfg["crouch"], lift=cfg["lift"], k_dcm=cfg["k_dcm"], k_step=cfg["k_step"],
                     ff=cfg["ff"], posture=cfg["posture"], head_fn=None, vel_fc=cfg["vel_fc"], pitch=cfg["pitch"],
                     dcm_to_com=cfg["dcm_to_com"], land_press=cfg["land_press"],
                     dyn_ff=cfg["dyn_ff"], lead_s=cfg["lead_s"], k_lat=cfg["k_lat"], lat_db=cfg["lat_db"], lat_in=cfg["lat_in"])
    nd = int(round(cfg["delay_ms"] / 1000 / dtc))
    buf = deque([d.ctrl.copy()] * (nd + 1), maxlen=nd + 1)
    # 놀이 관절 주소(액추에이터 순서)
    bl_q = np.array([ix.qadr.get("bl_" + j, -1) for j in ix.act_j]); bl_v = np.array([ix.vadr.get("bl_" + j, -1) for j in ix.act_j])
    has_bl = bl_q >= 0
    keys = ("com", "margin", "rpy", "fs", "u", "sat", "tau", "head_w", "crank_tau", "head_deg", "w_act", "clear", "contact", "bl")
    rec = {k: [] for k in keys}
    fall_t = None; tb = ix.torso
    n = int(T_end / m.opt.timestep)
    u_cmd = d.ctrl.copy(); cmd = d.ctrl.copy()
    wall0 = time.time()
    pad_half = {s: np.array([m.geom_size[g][1] for g in ix.pads[s]]) for s in "RL"}
    for k in range(n):
        t = k * m.opt.timestep
        if k % nsub == 0:
            sen = sense(m, d, ix)
            if has_bl.any():
                sen["q"][has_bl] += d.qpos[bl_q[has_bl]]; sen["qd"][has_bl] += d.qvel[bl_v[has_bl]]
            u_cmd = ctl.step(t, sen)
            buf.append(u_cmd)
            cmd = buf[0].copy()
        d.ctrl[:] = cmd
        if has_bl.any():
            d.ctrl[has_bl] = cmd[has_bl] - d.qpos[bl_q[has_bl]]     # 서보 P 제어가 출력각으로 닫힘
        mujoco.mj_step(m, d)
        R3 = d.xmat[tb].reshape(3, 3)
        tilt = math.degrees(math.acos(np.clip(R3[2, 2], -1, 1)))
        if fall_t is None and (tilt > 35 or d.subtree_com[tb][2] < 0.13):
            fall_t = t
        if fall_t is None and k % 5 == 0:
            cop, mg, Fz = true_cop_margin(m, d, ix)
            fs, u, sat = utilisation(m, d, ix)
            rec["com"].append(d.subtree_com[tb].copy()); rec["margin"].append(mg if mg is not None else np.nan)
            y, p, r_ = rpy_of(R3); rec["rpy"].append((r_, p, y))
            rec["fs"].append(fs); rec["u"].append(u); rec["sat"].append(sat); rec["tau"].append(d.actuator_force.copy())
            rec["w_act"].append(np.array([d.qvel[ix.vadr[j]] for j in ix.act_j]))
            w = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ix.head, w, 0)
            rec["head_w"].append(w[:3].copy()); rec["crank_tau"].append(0.0); rec["head_deg"].append(0.0)
            # 발 밑(패드 바닥 가장 낮은 점) 높이, 발마다 바닥 접촉 여부
            rec["clear"].append([float((d.geom_xpos[ix.pads[s], 2] - pad_half[s]).min()) for s in "RL"])
            con = {"R": False, "L": False}
            for ci in range(d.ncon):
                c = d.contact[ci]
                if ix.floor in (c.geom1, c.geom2):
                    g = c.geom2 if c.geom1 == ix.floor else c.geom1
                    if g in ix.pad_owner:
                        con[ix.pad_owner[g]] = True
            rec["contact"].append([con["R"], con["L"]])
            rec["bl"].append(np.where(has_bl, d.qpos[np.maximum(bl_q, 0)], 0.0))
        if fall_t is not None and t > fall_t + 0.5:
            break
    R = {k: np.array(v) for k, v in rec.items()}
    try:
        out = summarize(m, ix, cfg, plan, R, fall_t, T_end)
        out.update(extra_metrics(m, ix, cfg, R))
    except ValueError:                       # 시작 1 초 안에 넘어지면 요약할 구간이 없다
        out = {"fallen": fall_t is not None, "fall_t": fall_t, "T_end": T_end, "steps_done": 0, "early_fall": True}
    out["wall_s"] = time.time() - wall0
    return out, R, m


def extra_metrics(m, ix, cfg, R):
    """흔드는 발 최고 높이(걸음마다) · 들썩임 수 · 관절 부하 토크 RMS · 평균 · 최대(발열 · 강도 계산용)"""
    o = {}
    if not len(R["com"]):
        return o
    tt = np.arange(len(R["com"])) * 5e-3
    t_walk = cfg["t_walk"] + cfg["Tinit"]
    walk = tt >= t_walk
    peaks = []
    for si, s in enumerate("RL"):
        air = (~R["contact"][:, si]) & walk
        i = 0
        while i < len(air):
            if air[i]:
                j = i
                while j < len(air) and air[j]:
                    j += 1
                if (j - i) * 5e-3 > 0.03:
                    peaks.append((float(R["clear"][i:j, si].max()), (j - i) * 5e-3))
                i = j
            else:
                i += 1
    # 진짜 흔들기 = 바닥에서 0.3 s 이상 떨어진 구간(한 발 지지 0.45 s). 0.03–0.3 s 는 「들썩임」(디딤발이 잠깐 떴다 다시 닿음)
    sw = [p for p, dt in peaks if dt >= 0.3]; hop = [p for p, dt in peaks if dt < 0.3]
    o["swing_n"] = len(sw); o["hop_n"] = len(hop)
    o["swing_peaks_mm"] = [float(v * 1e3) for v in sw]
    o["hop_peaks_mm"] = [float(v * 1e3) for v in hop]
    o["swing_peak_mm"] = {"min": float(min(sw) * 1e3) if sw else None, "median": float(np.median(sw) * 1e3) if sw else None}
    b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
    load = R["tau"] - b * R["w_act"]
    sel = tt >= (t_walk if cfg["t_walk"] < 50 else 1.5)
    o["load_rms"] = {nm: float(np.sqrt((load[sel, i] ** 2).mean())) for i, nm in enumerate(ix.act)}
    o["load_mean_abs"] = {nm: float(np.abs(load[sel, i]).mean()) for i, nm in enumerate(ix.act)}
    o["load_max"] = {nm: float(np.abs(load[sel, i]).max()) for i, nm in enumerate(ix.act)}
    o["stall"] = {nm: float(ix.stall[i]) for i, nm in enumerate(ix.act)}
    if R["bl"].size:
        o["bl_absmax_deg"] = float(np.degrees(np.abs(R["bl"]).max()))
    return o


def one(job):
    meta, cfg = job
    try:
        o, R, m = run_rb(**cfg)
    except Exception as e:
        import traceback
        return meta, cfg, {"error": repr(e) + traceback.format_exc()[-600:]}
    return meta, cfg, o


def run_jobs(jobs, procs=20):
    from multiprocessing import Pool
    with Pool(min(procs, len(jobs))) as p:
        return p.map(one, jobs, chunksize=1)


def brief(o):
    if "error" in o:
        return "ERROR " + o["error"][-300:]
    mg = o.get("cop_margin_mm", {}); sw = o.get("swing_peak_mm", {})
    s = f"fall={o['fallen']}({o['fall_t']}) steps={o['steps_done']}"
    if mg.get("p5") is not None:
        s += f" p5={mg['p5']:.1f}mm"
    if sw.get("min") is not None:
        s += f" swingmin={sw['min']:.1f}mm"
    if "u_p99" in o:
        s += f" kneeU99={max(o['u_p99']['R_knee'], o['u_p99']['L_knee']):.2f} kneeSat={100 * max(o['sat_frac']['R_knee'], o['sat_frac']['L_knee']):.1f}%"
    s += f" wall={o.get('wall_s', 0):.0f}s"
    return s
