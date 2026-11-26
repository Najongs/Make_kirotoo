"""BMS 10 A 한계 판단: 넘어짐 · 일어나기처럼 여러 관절이 함께 크게 힘을 쓸 때 전체 서보 전류가 10 A 를 넘는 시간.

전류(스펙 방식 상한): I = I_대기 + |τ_부하|/Kt, 정지 전류에서 자름(PWM 제한 p 면 p × 정지 전류). [e-manual 11.1 V 값]
  τ_부하 = 액추에이터 힘 − 감쇠 b·ω. 전력 방식(배터리 쪽): P = V·I_대기 + (τ/Kt)²R + max(0, τω), R = V/I_정지 [추정 모델].
막는 방법 시뮬:
  - PWM Limit: 모든 서보의 최대 토크와 정지 전류를 p 배(forcerange × p). 레지스터 값 = p × 885.
  - Current Limit (XM 만): XM 의 최대 토크를 I_lim × Kt 로(forcerange). 레지스터 값 = I_lim / 2.69 mA.
  - 넘어짐 감지: 몸통 기울기 25° 를 넘으면 모든 서보 forcerange × 0.3 (PWM Limit 을 30 % 로 낮추는 것과 같음).
  - 일어나기 동작: 팔 · 다리를 동시에 vs 차례로.
모델: v18 + 고관절 롤 B(XM) + v20 서보 배치 + 등 뒤 250 g 한 팩(소유자 지정 가정값).
python battery6000/peak_current.py → battery6000/results/peak.json, peak_current.png
"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hiproll")); sys.path.insert(0, HERE)
import numpy as np, mujoco
from multiprocessing import Pool
from run_tests import BASE, push_t
from kw_common import build_model, scale_kp, Idx, jdump, korean_font
from sim_runs import CONFIGS

V = 11.1
EM = {3.8: (1.810, 2.1, 0.040), 1.75: (1.308, 1.3, 0.046), 1.4: (1.077, 1.3, 0.052)}   # forcerange → Kt, I_정지, I_대기
CFG = CONFIGS["등 뒤 250 g (한 팩)"]
LIMIT = 10.0


def currents(tau, w, fr0, b, p_scale):
    """tau, w: (n, nu). fr0: 원래 forcerange. p_scale: (nu,) 정지 전류 배수"""
    Kt = np.array([EM[round(f, 2)][0] for f in fr0]); Ist = np.array([EM[round(f, 2)][1] for f in fr0]); Isb = np.array([EM[round(f, 2)][2] for f in fr0])
    load = tau - b * w
    I = np.minimum(Isb + np.abs(load) / Kt, Ist * p_scale)
    Pw = V * Isb + (load / Kt) ** 2 * (V / Ist) + np.maximum(0.0, load * w)
    return I.sum(1), Pw.sum(1) / V


def stats(tot, dt=5e-3):
    def mov(x, n):
        if len(x) < n:
            return np.array([x.mean()])
        c = np.cumsum(np.insert(x, 0, 0)); return (c[n:] - c[:-n]) / n
    over = tot > LIMIT
    runs, r = [], 0
    for o in over:
        if o:
            r += 1
        elif r:
            runs.append(r); r = 0
    if r:
        runs.append(r)
    m20 = mov(tot, 4); m100 = mov(tot, 20); m1 = mov(tot, 200)
    return dict(max_5ms=float(tot.max()), max_20ms=float(m20.max()), max_100ms=float(m100.max()), max_1s=float(m1.max()),
                time_over_ms=float(over.sum() * dt * 1e3), longest_over_ms=float(max(runs) * dt * 1e3 if runs else 0.0),
                n_over_events=len(runs), time_20ms_avg_over_ms=float((m20 > LIMIT).sum() * dt * 1e3),
                time_100ms_avg_over_ms=float((m100 > LIMIT).sum() * dt * 1e3))


def sim_case(job):
    """kw_sim 걷기 · 넘어짐 경우"""
    name, extra = job
    from kw_sim import run
    m0, _ = build_model(hip_roll=CFG["hip_roll"], servo_map=CFG["servo_map"], extra_mass=CFG["extra_mass"])
    fr0 = m0.actuator_forcerange[:, 1].copy()
    b = np.array([m0.dof_damping[m0.jnt_dofadr[m0.actuator_trnid[i][0]]] for i in range(m0.nu)])
    c = dict(BASE); c.update(CFG); c.update(extra)
    pwm = extra.pop("_pwm", None) if False else c.pop("_pwm", None)
    xm_lim = c.pop("_xm_lim", None)
    o, R, m = run(**dict(c, _dummy=None)) if False else (None, None, None)
    # forcerange 를 바꾸는 경우는 run 이전에 모델을 고칠 수 없어서 extra_bodies 대신 kw_sim 의 fall_soft 와 별도 처리
    from kw_sim import run as run_
    import kw_sim
    p_scale = np.ones(len(fr0))
    if pwm or xm_lim:
        orig_build = kw_sim.build_model
        def patched(**kw):
            mm, sp = orig_build(**kw)
            fr = mm.actuator_forcerange[:, 1].copy()
            sc = np.ones(len(fr))
            if pwm:
                sc *= pwm
            if xm_lim:
                for i, f in enumerate(fr):
                    if abs(f - 3.8) < 1e-6:
                        sc[i] = min(sc[i], xm_lim * 1.810 / 3.8)
            mm.actuator_forcerange[:, 0] = -fr * sc; mm.actuator_forcerange[:, 1] = fr * sc
            return mm, sp
        kw_sim.build_model = patched
        if pwm:
            p_scale *= pwm
        if xm_lim:
            p_scale = np.where(np.abs(fr0 - 3.8) < 1e-6, np.minimum(p_scale, xm_lim / 2.1), p_scale)
    o, R, m = run_(**c)
    t5 = np.arange(len(R["tau"])) * 5e-3
    if c.get("record_fall") and o["fall_t"]:
        sel = (t5 >= o["fall_t"] - 1.0)
    else:
        sel = t5 >= 2.3
    ps = p_scale.copy()
    if c.get("fall_soft") and o.get("soft_t"):
        # 토크를 줄인 뒤 구간은 정지 전류도 줄어듦
        tot_hi, pw_hi = currents(R["tau"][sel], R["w_act"][sel], fr0, b, ps)
        after = t5[sel] >= o["soft_t"]
        tot_lo, pw_lo = currents(R["tau"][sel], R["w_act"][sel], fr0, b, ps * c["fall_soft"][1])
        tot = np.where(after, tot_lo, tot_hi); pw = np.where(after, pw_lo, pw_hi)
    else:
        tot, pw = currents(R["tau"][sel], R["w_act"][sel], fr0, b, ps)
    return dict(name=name, fallen=o["fallen"], fall_t=o["fall_t"], soft_t=o.get("soft_t"), speed=o.get("speed_mps"),
                margin_p5=o["cop_margin_mm"]["p5"], stats=stats(tot), stats_power=stats(pw),
                t=t5[sel][::2].tolist(), tot=tot[::2].tolist())


def getup(job):
    """엎드린 자세에서 열린 고리로 팔 · 다리를 밀어 올리는 동작(일어나기 흉내): 동시에 / 차례로, PWM 제한"""
    name, mode, pwm = job
    m, _ = build_model(hip_roll=CFG["hip_roll"], servo_map=CFG["servo_map"], extra_mass=CFG["extra_mass"])
    scale_kp(m, BASE["kp_scale"])
    fr0 = m.actuator_forcerange[:, 1].copy()
    if pwm:
        m.actuator_forcerange[:, 0] *= pwm; m.actuator_forcerange[:, 1] *= pwm
    b = np.array([m.dof_damping[m.jnt_dofadr[m.actuator_trnid[i][0]]] for i in range(m.nu)])
    d = mujoco.MjData(m); d.qpos[:] = m.qpos0
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]      # +x(앞)가 아래 = 엎드림
    d.qpos[2] = 0.11
    names = [m.actuator(i).name for i in range(m.nu)]
    tgt = np.zeros(m.nu)
    goal = {}
    for s in "RL":
        goal[f"{s}_arm_swing"] = math.radians(-100)
        goal[f"{s}_hip_pitch"] = math.radians(-60); goal[f"{s}_knee"] = math.radians(90); goal[f"{s}_ankle_pitch"] = math.radians(-30)
    T_settle, T_ramp = 1.0, 0.6
    def frac(t, t0):
        x = min(max((t - t0) / T_ramp, 0.0), 1.0); return x * x * (3 - 2 * x)
    taus, ws = [], []
    n = int(5.0 / m.opt.timestep)
    for k in range(n):
        t = k * m.opt.timestep
        for i, nm in enumerate(names):
            if nm in goal:
                if mode == "동시에":
                    f = frac(t, T_settle)
                else:   # 차례로: 팔 → 고관절 → 무릎 · 발목
                    t0 = T_settle + (0 if "arm" in nm else (0.8 if "hip_pitch" in nm else 1.6))
                    f = frac(t, t0)
                tgt[i] = goal[nm] * f
        d.ctrl[:] = tgt
        mujoco.mj_step(m, d)
        if k % 5 == 0 and t >= T_settle - 0.2:
            taus.append(d.actuator_force.copy()); ws.append(np.array([d.qvel[m.jnt_dofadr[m.actuator_trnid[i][0]]] for i in range(m.nu)]))
    taus = np.array(taus); ws = np.array(ws)
    tot, pw = currents(taus, ws, fr0, b, np.full(m.nu, pwm or 1.0))
    t5 = (np.arange(len(tot)) * 5e-3 + T_settle - 0.2)
    zt = float(d.subtree_com[1][2])
    return dict(name=name, stats=stats(tot), stats_power=stats(pw), com_z_end_m=zt, t=t5[::2].tolist(), tot=tot[::2].tolist())


if __name__ == "__main__":
    t0 = push_t(6, "ss")
    fall = dict(stride=0.03, push=(t0, 0.2, 15.0, "x"), T_end=t0 + 4, record_fall=True, after_fall_s=2.0)
    fall_side = dict(t_walk=99, T_end=10.0, push=(4.0, 0.2, 25.0, "y"), record_fall=True, after_fall_s=2.0)
    jobs = [("걷기 0.083 m/s", dict(stride=0.055)),
            ("걷기 0.083 m/s · PWM 70 %", dict(stride=0.055, _pwm=0.7)),
            ("걷기 0.076 m/s · PWM 70 %", dict(stride=0.05, _pwm=0.7)),
            ("걷기 0.076 m/s · XM 전류 제한 1.5 A", dict(stride=0.05, _xm_lim=1.5)),
            ("넘어짐: 걷다 앞으로 15 N", dict(fall)),
            ("넘어짐: 걷다 앞으로 15 N · 감지 후 30 %", dict(fall, fall_soft=(25.0, 0.3))),
            ("넘어짐: 걷다 앞으로 15 N · PWM 70 %", dict(fall, _pwm=0.7)),
            ("넘어짐: 걷다 앞으로 15 N · XM 전류 제한 1.5 A", dict(fall, _xm_lim=1.5)),
            ("넘어짐: 서 있다 옆 25 N", dict(fall_side)),
            ("넘어짐: 서 있다 옆 25 N · 감지 후 30 %", dict(fall_side, fall_soft=(25.0, 0.3)))]
    gjobs = [("일어나기 흉내: 동시에", "동시에", None), ("일어나기 흉내: 차례로", "차례로", None),
             ("일어나기 흉내: 동시에 · PWM 70 %", "동시에", 0.7), ("일어나기 흉내: 차례로 · PWM 70 %", "차례로", 0.7),
             ("일어나기 흉내: 동시에 · PWM 50 %", "동시에", 0.5)]
    with Pool(len(jobs) + len(gjobs)) as p:
        r1 = p.map_async(sim_case, jobs); r2 = p.map_async(getup, gjobs)
        rows = r1.get() + r2.get()
    jdump(rows, os.path.join(HERE, "results", "peak.json"))
    for r in rows:
        s = r["stats"]
        print(f"{r['name']:40s}", "fall" if r.get("fallen") else "", {k: round(v, 1) for k, v in s.items()}, "| power max20 %.1f" % r["stats_power"]["max_20ms"])
    plt = korean_font()
    fig, ax = plt.subplots(3, 1, figsize=(13, 10))
    for a, keys in zip(ax, (["넘어짐: 걷다 앞으로 15 N", "넘어짐: 걷다 앞으로 15 N · 감지 후 30 %", "넘어짐: 걷다 앞으로 15 N · PWM 70 %"],
                            ["일어나기 흉내: 동시에", "일어나기 흉내: 차례로", "일어나기 흉내: 동시에 · PWM 70 %"],
                            ["걷기 0.083 m/s", "걷기 0.083 m/s · PWM 70 %"])):
        for r in rows:
            if r["name"] in keys:
                a.plot(r["t"], r["tot"], lw=0.9, label=f"{r['name']} (10 A 넘는 시간 {r['stats']['time_over_ms']:.0f} ms)")
        a.axhline(LIMIT, color="k", ls="--", lw=1, label="BMS 최대 10 A")
        a.set_ylabel("서보 전류 합 상한 (A)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[-1].set_xlabel("시간 (s)")
    g = {r["name"]: r for r in rows}
    fig.suptitle(f"BMS 10 A 를 넘는 것은 넘어질 때 · 일어나기처럼 여러 관절이 한꺼번에 힘을 쓸 때뿐이다: 넘어짐 {g['넘어짐: 걷다 앞으로 15 N']['stats']['time_over_ms']:.0f} ms, "
                 f"일어나기(동시에) {g['일어나기 흉내: 동시에']['stats']['time_over_ms']:.0f} ms → 넘어짐 감지 후 토크 30 % 면 {g['넘어짐: 걷다 앞으로 15 N · 감지 후 30 %']['stats']['time_over_ms']:.0f} ms\n"
                 "(스펙 방식 상한 = 대기 전류 + 부하 토크/토크상수, e-manual 상수, 시뮬 실측 토크, v20 서보 배치 + 고관절 롤 B + 등 뒤 250 g 한 팩)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(HERE, "results", "peak_current.png"), dpi=90)
    print("saved")
