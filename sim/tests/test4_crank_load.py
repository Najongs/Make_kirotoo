"""Test 4: XL430 head crank (nod_crank) load.
 - during the gait inverse dynamics (test 2) and the open-loop walk (test 3): read from their json;
 - torso fixed, head nodding -10 -> +20 -> -10 deg once per 1 s and once per 0.5 s:
   (a) inverse dynamics need: crank dof + head dof * dbeta/dalpha + ear dof * dpsi/dalpha (rigid, and + armature);
   (b) simulated position actuator force (crank servo tracks alpha(beta(t))).
Run: cd sim && uv run python tests/test4_crank_load.py  -> results/test4_crank_load.json, .png
"""
import math, json, numpy as np
from common import *

mf, df = load(fixed=True)
STALL, W0 = SPEC_MODEL["XL430-W250"]; W0E = SPEC_EMAN["XL430-W250"][1]
A_T, B_T = np.radians(LOOP["crank_deg"]), np.radians(LOOP["beta_deg"])
CF_AB = np.polyfit(B_T, A_T, 6)          # smooth alpha(beta): piecewise-linear interp would add noise to alpha''
alpha_of = lambda b: np.polyval(CF_AB, b)
out = {}
for key in ("test2_gait_torque.json", "test3_openloop_walk.json"):
    p = os.path.join(RES, key)
    if os.path.exists(p):
        j = json.load(open(p))
        if "cases" in j:
            out["gait_inverse_crank_load_Nm"] = {k: v["nod_crank_load_max"] for k, v in j["cases"].items()}
        else:
            out["openloop_walk_crank_force_Nm"] = {k: j[k]["nod_crank_force_max"] for k in ("kp_x1", "kp_x4")}
series = {}
for P in (1.0, 0.5):
    dt = mf.opt.timestep
    ts = np.arange(-2, int(round(3 * P / dt)) + 3) * dt
    beta = np.radians(5.0 - 15.0 * np.cos(2 * np.pi * ts / P))
    alpha = alpha_of(beta)
    res = {}; keep = {}
    for arm in (False, True):
        m = rigid_copy(mf, keep_armature=arm); d = mujoco.MjData(m)
        Q = []
        for a in alpha:
            q = m.qpos0.copy(); set_joints(m, q, {"nod_crank": a}); Q.append(q)
        Q = np.array(Q); V, A = fd(m, Q, dt)
        need = []
        for k in range(len(V)):
            d.qpos[:] = Q[k + 1]; d.qvel[:] = V[k]; d.qacc[:] = A[k]; mujoco.mj_inverse(m, d)
            fH, fE = loop_slopes(Q[k + 1][qa(m, "nod_crank")])
            need.append(d.qfrc_inverse[va(m, "nod_crank")] + d.qfrc_inverse[va(m, "head_nod")] * fH + d.qfrc_inverse[va(m, "ear_axle")] * fE)
        need = np.array(need); w = V[:, va(m, "nod_crank")]
        res["armature" if arm else "rigid"] = {"tau_max": float(np.abs(need).max()), "w_max": float(np.abs(w).max()),
                                                "u": float(np.max(np.abs(need) / STALL + np.abs(w) / W0)),
                                                "u_emanual": float(np.max(np.abs(need) / STALL + np.abs(w) / W0E))}
        keep[arm] = (need, w)
        if arm:
            series[P] = (ts[1:-1], need, w)
    # u is linear in the armature: find the XL armature where u = 1 (bisection on the scale)
    (nr, w_), (na, _) = keep[False], keep[True]
    def u_of(sc):
        return float(np.max(np.abs(nr + sc * (na - nr)) / STALL + np.abs(w_) / W0))
    lo, hi = 0.0, 50.0
    if u_of(0) >= 1:
        be = 0.0
    elif u_of(hi) < 1:
        be = None
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if u_of(mid) < 1 else (lo, mid)
        be = lo * float(mf.dof_armature[va(mf, "nod_crank")])
    res["u_vs_armature_scale"] = {str(sc): u_of(sc) for sc in (0.25, 0.5, 1.0)}
    res["xl_armature_for_u1_kgm2"] = be
    # (b) position actuator in the full fixed model
    m, d = load(fixed=True); mujoco.mj_resetDataKeyframe(m, d, 0)
    ai = aid(m, "nod_crank"); n = int(round(3 * P / m.opt.timestep))
    f = []; err = []; hb = []
    for k in range(n):
        t = k * m.opt.timestep
        b = math.radians(5.0 - 15.0 * math.cos(2 * math.pi * t / P))
        d.ctrl[ai] = alpha_of(b); mujoco.mj_step(m, d)
        if t > P:
            f.append(d.actuator_force[ai]); hb.append(d.qpos[qa(m, "head_nod")]); err.append(b - d.qpos[qa(m, "head_nod")])
    hb = np.degrees(hb)
    res["sim_position_actuator"] = {"force_max": float(np.abs(f).max()), "force_over_stall": float(np.abs(f).max() / STALL),
                                    "head_angle_reached_deg": [float(hb.min()), float(hb.max())],
                                    "head_tracking_error_max_deg": float(np.degrees(np.abs(err).max())),
                                    "saturated_frac": float(np.mean(np.abs(f) >= STALL * 0.999))}
    out[f"nod_period_{P}s"] = res
    print(P, res)
jdump(out, "test4_crank_load.json")
plt = korean_font()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, P in zip(ax, (1.0, 0.5)):
    t, need, w = series[P]
    a.plot(t, need, color="#2a6fdb", lw=2, label="필요 토크 (역동역학, 회전자 관성 포함)")
    a.axhline(STALL, color="#c0392b", ls="--", label="XL430 정지토크 1.4 N·m"); a.axhline(-STALL, color="#c0392b", ls="--")
    a.set_title(f"머리 −10→+20→−10° 를 {P:g} 초에 한 번", fontsize=11); a.set_xlabel("시간 (s)"); a.set_ylabel("크랭크 토크 (N·m)")
    a.grid(alpha=0.3); a.legend(fontsize=8)
r1, r5 = out["nod_period_1.0s"], out["nod_period_0.5s"]
fig.suptitle(f"끄덕임 크랭크(XL430) 사용률: 1초 주기 {r1['armature']['u']:.2f}, 0.5초 주기 {r5['armature']['u']:.2f} "
             f"(회전자 관성 [추정] 포함; 관성 빼면 {r1['rigid']['u']:.2f} · {r5['rigid']['u']:.2f}), 걷는 중 부하는 {max(out.get('openloop_walk_crank_force_Nm', {'x': 0}).values()):.2f} N·m 이하", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(RES, "test4_crank_load.png"), dpi=110)
print("saved")
