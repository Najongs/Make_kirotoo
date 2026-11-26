"""Test 1: servo_study.static_cases poses (single support on the right foot: crouch 40 mm, CoM over the toe, over the
heel; body rolled on the stance ankle so the CoM is over the stance foot laterally).
 (a) mj_inverse with the stance foot as root (gravity only) vs servo_study joint torques;
 (b) free model, position actuators holding the pose for 3 s: fall?, max |force| / stall.

Run: cd sim && uv run python tests/test1_static_hold.py   -> results/test1_static_hold.json, .png
"""
import math, numpy as np
from common import *
from scipy.spatial import ConvexHull

m_free, d_free = load()
m = rigid_copy(m_free); d = mujoco.MjData(m)
M = ss_model(False)
ss_static = S.static_cases(M)
torso_b = bid(m_free, "torso")
pads = [i for i in range(m_free.ngeom) if "_pad" in gname(m_free, i)]


def static_pose(crouch, xr):
    """same iteration as servo_study.static_cases"""
    ys = -40.0; H = 150.0 - crouch
    q1, q2 = S.leg_ik(-xr, -H)
    psi_ = 0.0
    for _ in range(30):
        R = S.rot("x", psi_)
        a_pel = np.array([-xr, ys, 185.0 - H])
        P = np.array([0.0, ys, 35.0]) - R @ a_pel
        q_st = (0.0, q1, q2, -(q1 + q2), -psi_)
        bod, jts = S.poses(M, ys, q_st, q_st, P, R, (0, 0))
        mm = sum(M.L[b]["mass_g"] for b in bod)
        cy = sum(M.L[b]["mass_g"] * bod[b][1][1] for b in bod) / mm
        psi_ += (cy - ys) / 190.0
    return q_st, psi_


def hull_margin(pts, p):
    H = ConvexHull(pts)
    return min(-(eq[:2] @ p + eq[2]) for eq in H.equations)


def hold(q, ang, kp_scale=1.0, snap_key=None, ramp=1.0, pad_solref=None):
    """free model, position targets = pose. Gravity ramps 0 -> 1 g over `ramp` s (quasi-static loading, so the
    servos are not hit by the full weight in one step), then 3 s at 1 g."""
    mm = mujoco.MjModel.from_xml_path(XML); dd = mujoco.MjData(mm)
    mm.actuator_gainprm[:, 0] *= kp_scale; mm.actuator_biasprm[:, 1] *= kp_scale
    if pad_solref:
        for i in range(mm.ngeom):
            if "_pad" in gname(mm, i):
                mm.geom_solref[i] = pad_solref
    dd.qpos[:] = q; dd.qpos[2] += 0.0003
    ctrl = np.zeros(mm.nu)
    for i in range(mm.nu):
        jn = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, mm.actuator_trnid[i][0])
        ctrl[i] = ang.get(jn, 0.0)
    mujoco.mj_forward(mm, dd)
    z0 = dd.subtree_com[torso_b][2]; R0 = dd.xmat[torso_b].reshape(3, 3).copy()
    fmax = np.zeros(mm.nu); tilt = 0.0; drop = 0.0; lfoot_touch = False; t_fall = None
    g0 = mm.opt.gravity.copy(); nr = int(ramp / mm.opt.timestep)
    for k in range(nr + 3000):
        mm.opt.gravity[:] = g0 * min(1.0, (k + 1) / nr) if nr else g0
        dd.ctrl[:] = ctrl; mujoco.mj_step(mm, dd)
        if k > 50:
            fmax = np.maximum(fmax, np.abs(dd.actuator_force))
        Rt = dd.xmat[torso_b].reshape(3, 3)
        tl = math.degrees(math.acos(np.clip((np.trace(R0.T @ Rt) - 1) / 2, -1, 1)))
        tilt = max(tilt, tl); drop = max(drop, (z0 - dd.subtree_com[torso_b][2]) * 1e3)
        if t_fall is None and (tl > 10 or drop > 30):
            t_fall = k * mm.opt.timestep - ramp
        for c in dd.contact[:dd.ncon]:
            names = (body_of_geom(mm, c.geom1), body_of_geom(mm, c.geom2))
            if "world" in names and any(n and n.startswith("L_") for n in names):
                lfoot_touch = True
        if k == nr + 1500 and snap_key:
            snaps[snap_key] = dd.qpos.copy()
    return {"kp_scale": kp_scale, "gravity_ramp_s": ramp, "pad_solref": pad_solref, "fallen": t_fall is not None, "fall_time_s": t_fall, "max_rotation_from_start_deg": tilt,
            "max_com_drop_mm": drop, "left_leg_touched_floor": lfoot_touch,
            "force_over_stall": {mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_ACTUATOR, i): float(fmax[i] / mm.actuator_forcerange[i][1])
                                 for i in range(mm.nu)}}


R_PADS, R_SOLE = foot_support(m_free, "R")
out = {"cases": {}}
snaps = {}
for nm, crouch, xr in (("crouch", 40.0, 0.0), ("toe", 15.0, 55.0), ("heel", 15.0, -60.0)):
    q_st, psi = static_pose(crouch, xr)
    ang = {}
    for s in "RL":
        ang.update({f"{s}_{j}": q_st[i] for i, j in enumerate(LEGJ)})
    q = torso_for_stance(m, d, ang, "R")
    tau, w, cop = inverse_foot_root(m, d, q, np.zeros(m.nv), np.zeros(m.nv), "R")
    tqs, zmp_ss, psi_ss = ss_static[nm]
    rows = {}
    for j in LEGJ:
        rows[j] = {"mj": float(abs(tau[va(m, f"R_{j}")])), "servo_study": float(tqs[f"stance_{j}"]),
                   "swing_leg_mj": float(abs(tau[va(m, f"L_{j}")]))}
    # CoM vs support
    d_free.qpos[:] = q; mujoco.mj_forward(m_free, d_free)
    com = d_free.subtree_com[torso_b] * 1e3
    res = {"psi_deg": float(np.degrees(psi)), "psi_servo_study_deg": float(psi_ss), "joint_torque": rows,
           "cop_mj_mm": [float(cop[0] * 1e3), float(cop[1] * 1e3)], "zmp_servo_study_mm": [float(zmp_ss[0]), float(zmp_ss[1])],
           "com_mm": com.tolist(), "com_margin_pads_mm": float(hull_margin(R_PADS, com[:2])),
           "com_margin_sole_mm": float(hull_margin(R_SOLE, com[:2]))}
    # (b) hold in the free model; static joint deflection = tau / kp
    res["static_deflection_deg"] = {j: float(np.degrees(rows[j]["mj"] / {"XM430-W350": 17.5, "XC430-W240": 7.0}[SERVO_OF[j]]))
                                    for j in LEGJ}
    res["hold"] = hold(q, ang, 1.0, nm)
    res["hold_kp_x4"] = hold(q, ang, 4.0, nm + "_x4")
    res["hold_sudden_load"] = hold(q, ang, 1.0, None, ramp=0.0)
    res["hold_kp_x4_stiff_pads"] = hold(q, ang, 4.0, None, pad_solref=[0.005, 1.0])
    out["cases"][nm] = res
    print(nm, f"psi {res['psi_deg']:.2f} (ss {psi_ss:.2f})", {j: (round(r['mj'], 3), round(r['servo_study'], 3)) for j, r in rows.items()})
    print("   cop", np.round(res["cop_mj_mm"], 1), "zmp ss", np.round(res["zmp_servo_study_mm"], 1), "com", np.round(com, 1),
          f"margin pads {res['com_margin_pads_mm']:.1f} sole {res['com_margin_sole_mm']:.1f}")
    for h in (res["hold"], res["hold_kp_x4"], res["hold_sudden_load"], res["hold_kp_x4_stiff_pads"]):
      print("   kp x", h["kp_scale"], "ramp", h["gravity_ramp_s"], "pads", h["pad_solref"], "fallen", h["fallen"], h["fall_time_s"], f"rot {h['max_rotation_from_start_deg']:.1f} drop {h['max_com_drop_mm']:.1f}",
          "L touch", h["left_leg_touched_floor"], "max f/stall", round(max(h["force_over_stall"].values()), 3),
          max(h["force_over_stall"], key=h["force_over_stall"].get))
    print("   static deflection deg", {k: round(v, 1) for k, v in res["static_deflection_deg"].items()})
jdump(out, "test1_static_hold.json")

# figure: snapshots at 1.5 s
plt = korean_font()
ren = mujoco.Renderer(m_free, 480, 400)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
fig, axs = plt.subplots(2, 3, figsize=(12, 9.5))
for a, (nm, key, hk) in zip(axs.ravel(), [(n, n + sfx, hk) for sfx, hk in (("", "hold"), ("_x4", "hold_kp_x4")) for n in ("crouch", "toe", "heel")]):
    d_free.qpos[:] = snaps[key]; mujoco.mj_forward(m_free, d_free)
    cam.lookat[:] = [0.0, -0.02, 0.2]; cam.distance = 1.0; cam.azimuth = 215; cam.elevation = -12
    ren.update_scene(d_free, cam); a.imshow(ren.render()); a.axis("off")
    h = out["cases"][nm][hk]; r = out["cases"][nm]
    a.set_title(f"{ {'crouch': '한 발 앉기 40 mm', 'toe': '발끝 위 무게중심', 'heel': '뒤꿈치 위 무게중심'}[nm]}\n"
                f"위치 이득 ×{h['kp_scale']:.0f}: {('넘어짐 (중력 키우는 중)' if h['fall_time_s'] < 0 else '넘어짐 (1 g 후 %.2f s)' % h['fall_time_s']) if h['fallen'] else '3 s 버팀'} · 무게중심 여유(패드) {r['com_margin_pads_mm']:.0f} mm",
                fontsize=10)
nf = sum(out["cases"][k]["hold"]["fallen"] for k in out["cases"]); nf4 = sum(out["cases"][k]["hold_kp_x4"]["fallen"] for k in out["cases"])
fig.suptitle(f"한 발 정지 자세 3가지 중 추정 기본 이득(XM 17.5 · XC 7.0 N·m/rad)으로는 {nf}개, 이득 4배로는 {nf4}개가 3초 안에 넘어진다\n"
             f"(위: 기본 이득, 아래: 4배; 중력을 1초에 걸쳐 0→1 g 로 키운 뒤 1.5 s 장면, 시뮬 실측, 오른발 디딤)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(RES, "test1_static_hold.png"), dpi=110)
print("saved")
