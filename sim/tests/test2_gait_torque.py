"""Test 2: servo_study.gait() joint trajectories (walk / fast / jog, psi from tune_psi) through MuJoCo mj_inverse.

Stance foot = root (servo_study assumption: stance foot flat and fixed on the floor). For each sample the torso pose
is solved from the stance foot (forward kinematics), qvel/qacc by central differences, mj_inverse gives the
generalized force; its 6 free-joint components are taken by a ground wrench on the stance foot (J^T w), the rest
are the joint torques. Compared with servo_study.dynamics() on the same samples.

Run: cd sim && uv run python tests/test2_gait_torque.py   -> results/test2_gait_torque.json, .png
"""
import math, numpy as np
from common import *

m_full, d_full = load()
m = rigid_copy(m_full); d = mujoco.MjData(m)
m_full = rigid_copy(m_full, keep_armature=True); d_full = mujoco.MjData(m_full)
cases, psi_info = tuned_cases()
M_true = ss_model(False); M_opt = ss_model(True)
N = 200
ROLE = {"stance": "R", "swing": "L"}
out = {"psi_deg": {k: v[0] for k, v in psi_info.items()}, "zmp_err_mm": {k: v[1] for k, v in psi_info.items()},
       "method": "stance foot as root: torso pose from FK so the R foot stays flat at its zero-pose place; "
                 "central differences; mj_inverse on a copy without damping/constraints, with and without armature; free-joint "
                 "components mapped to a ground wrench on R_foot via its body Jacobian", "cases": {}}
SERIES = {}; NPZ = {}
for nm, case in cases.items():
    T = case["T"]; dt = T / N
    ts = np.arange(-2, N + 3) * dt
    rec, dt_ss = S.gait(M_true, n=N, **case)
    assert abs(dt_ss - dt) < 1e-12
    Q = []; chk_q = 0.0; chk_p = 0.0
    for k, t in enumerate(ts):
        ang, psi = gait_angles(case, t / T)
        q = torso_for_stance(m, d, ang, "R")
        Q.append(q)
        # sanity: same joint angles and torso pose as servo_study
        st, sw = rec[k][3]["st"], rec[k][3]["sw"]
        chk_q = max(chk_q, max(abs(ang[f"R_{j}"] - st[i]) for i, j in enumerate(LEGJ)),
                    max(abs(ang[f"L_{j}"] - sw[i]) for i, j in enumerate(LEGJ)))
        Rss, cww = rec[k][1]["torso"]
        P_ss = cww - Rss @ np.array(M_true.L["torso"]["com_world"])
        chk_p = max(chk_p, np.linalg.norm(q[:3] * 1e3 - P_ss))
    Q = np.array(Q)
    V, A = fd(m, Q, dt)
    K = len(V)
    TAU = np.zeros((K, m.nv)); TAUF = np.zeros((K, m.nv)); COP = np.zeros((K, 3)); W = np.zeros((K, 6))
    for k in range(K):
        TAU[k], W[k], COP[k] = inverse_foot_root(m, d, Q[k + 1], V[k], A[k], "R")
        TAUF[k], _, _ = inverse_foot_root(m_full, d_full, Q[k + 1], V[k], A[k], "R")
    # servo_study on the same samples
    ss = {}
    for tag, MM in (("true_mass", M_true), ("servo_opt_mass", M_opt)):
        rr, _ = S.gait(MM, n=N, **case)
        tq, qs, zmp = S.dynamics(MM, rr, dt)
        ss[tag] = (tq, qs, zmp)
    res = {"check_joint_angle_diff_rad": chk_q, "check_torso_pos_diff_mm": chk_p, "joints": {}}
    for role, side in ROLE.items():
        for j in LEGJ:
            dof = va(m, f"{side}_{j}")
            tau = TAU[:, dof]; tauf = TAUF[:, dof]; w = V[:, dof]
            st, w0 = SPEC_MODEL[SERVO_OF[j]]; ste, w0e = SPEC_EMAN[SERVO_OF[j]]
            r = {"mj_tau_max": float(np.abs(tau).max()), "mj_tau_max_with_armature": float(np.abs(tauf).max()),
                 "mj_w_max": float(np.abs(w).max()),
                 "mj_u": float(np.max(np.abs(tau) / st + np.abs(w) / w0)),
                 "mj_u_emanual": float(np.max(np.abs(tau) / ste + np.abs(w) / w0e)),
                 "mj_u_with_armature": float(np.max(np.abs(tauf) / st + np.abs(w) / w0)),
                 "mj_u_with_armature_emanual": float(np.max(np.abs(tauf) / ste + np.abs(w) / w0e))}
            for tag, (tq, qs, zmp) in ss.items():
                t2, w2 = tq[f"{role}_{j}"], qs[f"{role}_{j}"]
                r[f"ss_{tag}_tau_max"] = float(np.abs(t2).max()); r[f"ss_{tag}_w_max"] = float(np.abs(w2).max())
                r[f"ss_{tag}_u"] = float(np.max(np.abs(t2) / st + np.abs(w2) / w0))
            res["joints"][f"{role}_{j}"] = r
    # arms and the head crank (XL430): crank load = own dof + head dof * dbeta/dalpha + ear dof * dpsi/dalpha
    fH, fE = loop_slopes(0.0)
    crank = TAU[:, va(m, "nod_crank")] + TAU[:, va(m, "head_nod")] * fH + TAU[:, va(m, "ear_axle")] * fE
    res["nod_crank_load_max"] = float(np.abs(crank).max())
    res["head_nod_dof_max"] = float(np.abs(TAU[:, va(m, "head_nod")]).max())
    for s in "RL":
        dof = va(m, f"{s}_arm_swing"); st, w0 = SPEC_MODEL["XC430-W240"]
        res[f"{s}_arm_swing"] = {"mj_tau_max": float(np.abs(TAU[:, dof]).max()), "mj_w_max": float(np.abs(V[:, dof]).max()),
                                 "mj_u": float(np.max(np.abs(TAU[:, dof]) / st + np.abs(V[:, dof]) / w0))}
    # centre of pressure vs the foot: pads hull (x -68..54, y -69..-11) and the full sole used by servo_study
    zss = ss["true_mass"][2]
    res["cop_mj_x_mm"] = [float(COP[:, 0].min() * 1e3), float(COP[:, 0].max() * 1e3)]
    res["cop_mj_y_mm"] = [float(COP[:, 1].min() * 1e3), float(COP[:, 1].max() * 1e3)]
    res["zmp_ss_x_mm"] = [float(zss[:, 0].min()), float(zss[:, 0].max())]
    res["zmp_ss_y_mm"] = [float(zss[:, 1].min()), float(zss[:, 1].max())]
    res["zmp_max_diff_mm"] = float(np.max(np.linalg.norm(COP[:, :2] * 1e3 - zss, axis=1)))
    PADH, SOLEH = foot_support(m_full, "R")
    mp = np.array([hull_margin(PADH, c * 1e3) for c in COP]); ms_ = np.array([hull_margin(SOLEH, c * 1e3) for c in COP])
    res["cop_outside_pads_frac"] = float(np.mean(mp < 0))
    res["cop_outside_sole_frac"] = float(np.mean(ms_ < 0))
    res["ground_force_z_N"] = [float(W[:, 2].min()), float(W[:, 2].max())]
    out["cases"][nm] = res
    SERIES[nm] = (ts[1:-1], TAU, V, ss["true_mass"][0], COP, zss)
    NPZ[nm] = (TAU, TAUF, V)
    print(nm, f"angle check {chk_q:.1e} rad, torso pos check {chk_p:.2e} mm, zmp diff {res['zmp_max_diff_mm']:.2f} mm")
    for k, r in res["joints"].items():
        print(f"  {k:20s} tau mj {r['mj_tau_max']:.3f} ss {r['ss_true_mass_tau_max']:.3f} ss(opt) {r['ss_servo_opt_mass_tau_max']:.3f}"
              f" | w mj {r['mj_w_max']:.2f} ss {r['ss_true_mass_w_max']:.2f} | u mj {r['mj_u']:.3f} ss {r['ss_true_mass_u']:.3f}"
              f" | +armature {r['mj_tau_max_with_armature']:.3f}")
    print("  crank load", res["nod_crank_load_max"], "arms", res["R_arm_swing"], "cop x", res["cop_mj_x_mm"], "y", res["cop_mj_y_mm"],
          "outside pads", res["cop_outside_pads_frac"], "outside sole", res["cop_outside_sole_frac"])
# ---------------- servo_opt.hip_strategy: single support on R (crouch 15, roll 12.5 deg), the upper body swings
# about the stance hip-pitch axis, theta = 8 deg sin(2 pi t / 0.5 s): stance hip pitch = q1 - theta, all else fixed
import servo_opt as O
A_deg, Tp, nh = 8.0, 0.5, 120
H = 150.0 - 15.0; q1, q2 = S.leg_ik(0.0, -H); psi = np.radians(12.5); dt = Tp / nh
Q = []
for k in range(-2, nh + 3):
    th = np.radians(A_deg) * np.sin(2 * np.pi * k * dt / Tp)
    ang = {}
    for sd, hp in (("R", q1 - th), ("L", q1)):
        ang.update({f"{sd}_hip_yaw": 0.0, f"{sd}_hip_pitch": hp, f"{sd}_knee": q2, f"{sd}_ankle_pitch": -(q1 + q2), f"{sd}_ankle_roll": -psi})
    Q.append(torso_for_stance(m, d, ang, "R"))
Q = np.array(Q); V, A = fd(m, Q, dt)
th_mj = np.array([inverse_foot_root(m, d, Q[k + 1], V[k], A[k], "R")[0][va(m, "R_hip_pitch")] for k in range(len(V))])
th_mja = np.array([inverse_foot_root(m_full, d_full, Q[k + 1], V[k], A[k], "R")[0][va(m, "R_hip_pitch")] for k in range(len(V))])
w_h = V[:, va(m, "R_hip_pitch")]
hs = O.hip_strategy(M_true)["stance_hip_pitch"]
st, w0 = SPEC_MODEL["XM430-W350"]; ste, w0e = SPEC_EMAN["XM430-W350"]
out["hip_strategy"] = {"mj_tau_max": float(np.abs(th_mj).max()), "mj_tau_max_with_armature": float(np.abs(th_mja).max()),
                       "mj_w_max": float(np.abs(w_h).max()), "ss_tau_max": float(np.abs(hs[0]).max()), "ss_w_max": float(np.abs(hs[1]).max()),
                       "mj_u": float(np.max(np.abs(th_mj) / st + np.abs(w_h) / w0)), "mj_u_emanual": float(np.max(np.abs(th_mj) / ste + np.abs(w_h) / w0e)),
                       "mj_u_with_armature": float(np.max(np.abs(th_mja) / st + np.abs(w_h) / w0)),
                       "mj_u_with_armature_emanual": float(np.max(np.abs(th_mja) / ste + np.abs(w_h) / w0e))}
NPZ["hip_strategy"] = (th_mj[:, None], th_mja[:, None], w_h[:, None])
print("hip_strategy", out["hip_strategy"])
np.savez(os.path.join(RES, "test2_series.npz"), **{f"{k}_{n}": v for k, (a, b, c) in NPZ.items() for n, v in (("tau_rigid", a), ("tau_armature", b), ("qvel", c))},
         dof_names=np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) or "" for i in range(m.nv)]))
jdump(out, "test2_gait_torque.json")

# ---------------- figure: walk time series of the loaded joints, MuJoCo vs servo_study
plt = korean_font()
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
t, TAU, V, tq, COP, zss = SERIES["walk"]
items = [("stance", "knee"), ("stance", "ankle_pitch"), ("stance", "ankle_roll"), ("stance", "hip_pitch"),
         ("swing", "knee"), ("stance", "hip_yaw")]
KR = {"hip_yaw": "고관절 요", "hip_pitch": "고관절 피치", "knee": "무릎", "ankle_pitch": "발목 피치", "ankle_roll": "발목 롤"}
for a, (role, j) in zip(ax.ravel(), items):
    dof = va(m, f"{ROLE[role]}_{j}")
    a.plot(t * 1e3, np.abs(TAU[:, dof]), color="#2a6fdb", lw=2, label="MuJoCo mj_inverse (시뮬 실측)")
    a.plot(t * 1e3, np.abs(tq[f"{role}_{j}"]), color="#e07b39", lw=1.6, ls="--", label="servo_study (설계 코드)")
    a.set_title(f"{'디딤' if role == 'stance' else '흔드는'} 다리 {KR[j]}", fontsize=11)
    a.set_ylabel("|토크| (N·m)"); a.grid(alpha=0.3)
ax[1, 0].set_xlabel("한 걸음 안의 시간 (ms)"); ax[1, 1].set_xlabel("한 걸음 안의 시간 (ms)"); ax[1, 2].set_xlabel("한 걸음 안의 시간 (ms)")
ax[0, 0].legend(fontsize=9)
wk = out["cases"]["walk"]["joints"]
worst = max(abs(r["mj_tau_max"] - r["ss_true_mass_tau_max"]) for r in wk.values())
fig.suptitle(f"보통 걸음(walk: 보폭 30 mm, 0.45 s/걸음, 몸 기울임 {out['psi_deg']['walk']}°)에서 MuJoCo 역동역학과 servo_study 관절 토크가 겹친다 "
             f"— 관절별 최대값 차이 최대 {worst:.3f} N·m", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(RES, "test2_gait_torque.png"), dpi=110)
print("saved")
