"""Test 3: open-loop replay of the servo_study 'walk' joint trajectory as position targets on the free model.

Timeline: 0.5 s stand (zero pose) -> 1.0 s crouch to the walk crouch -> 2 ramp steps (step / roll / lift / arm
amplitude 0 -> 1) -> 10 full steps. Stance alternates R, L (L = mirror: roll joints and torso roll negated).
No feedback of any kind. Records falls, foot slip, foot-foot contact, actuator force / stall; video + frames.

Run: cd sim && uv run python tests/test3_openloop_walk.py  -> results/test3_openloop_walk.json, .mp4, .png
"""
import math, numpy as np
from common import *

cases, psi_info = tuned_cases()
CASE = cases["walk"]; T = CASE["T"]
T_STAND, T_CROUCH, N_RAMP, N_FULL = 0.5, 1.0, 2, 10
T_END = T_STAND + T_CROUCH + (N_RAMP + N_FULL) * T


def smooth(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


def targets(t):
    """joint targets (rad) at time t, and (step index, stance) or None"""
    if t < T_STAND:
        return {}, None
    if t < T_STAND + T_CROUCH:
        return crouch_angles(CASE["crouch"] * smooth((t - T_STAND) / T_CROUCH)), None
    tw = t - T_STAND - T_CROUCH
    k = int(tw // T); u = tw / T - k
    scale = smooth(tw / (N_RAMP * T))
    stance = "R" if k % 2 == 0 else "L"
    ang, _ = gait_angles(CASE, u, scale, stance)
    return ang, (k, stance)


def run(kp_scale=1.0, video=None, frames_at=()):
    m, d = load()
    m.actuator_gainprm[:, 0] *= kp_scale; m.actuator_biasprm[:, 1] *= kp_scale
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    act_j = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]) for i in range(m.nu)]
    tb = bid(m, "torso"); dt = m.opt.timestep
    fmax = np.zeros(m.nu); fall_t = None; fall_step = None; ff_contacts = []; first_floor_nonpad = None
    slip = {"R": 0.0, "L": 0.0}; touch0 = {"R": None, "L": None}
    track = []
    ren = cam = None; frames = []; shots = {}
    if video or frames_at:
        ren = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    n = int(T_END / dt); fps = 30; every = int(round(1 / (fps * dt)))
    crank_f = 0.0
    pads = {sd: [i for i in range(m.ngeom) if gname(m, i).startswith(f"{sd}_foot_pad")] for sd in "RL"}
    FS = []; supp = []; clear = {"R": 0.0, "L": 0.0}; yaw = []
    t_full = T_STAND + T_CROUCH + N_RAMP * T
    for k in range(n):
        t = k * dt
        ang, st = targets(t)
        for i, jn in enumerate(act_j):
            d.ctrl[i] = ang.get(jn, 0.0)
        mujoco.mj_step(m, d)
        if t > 0.05:
            fmax = np.maximum(fmax, np.abs(d.actuator_force))
        crank_f = max(crank_f, abs(d.actuator_force[aid(m, "nod_crank")]))
        R3 = d.xmat[tb].reshape(3, 3); tilt = math.degrees(math.acos(np.clip(R3[2, 2], -1, 1)))
        comz = d.subtree_com[tb][2]
        if fall_t is None and (tilt > 30 or comz < 0.15):
            fall_t = t; fall_step = st
        # contacts
        foot_on = {"R": False, "L": False}
        for c in d.contact[:d.ncon]:
            b1, b2 = body_of_geom(m, c.geom1), body_of_geom(m, c.geom2)
            if "world" in (b1, b2):
                other = b2 if b1 == "world" else b1; g = gname(m, c.geom2 if b1 == "world" else c.geom1)
                if other.endswith("_foot") and "_pad" in g:
                    foot_on[other[0]] = True
                elif first_floor_nonpad is None:
                    first_floor_nonpad = (round(t, 3), g)
            elif b1[0] in "RL" and b2[0] in "RL" and b1[0] != b2[0]:
                ff_contacts.append((round(t, 3), gname(m, c.geom1), gname(m, c.geom2)))
        for s in "RL":
            p = d.xpos[bid(m, f"{s}_foot")][:2].copy()
            if foot_on[s]:
                if touch0[s] is None:
                    touch0[s] = p
                elif fall_t is None:
                    slip[s] = max(slip[s], float(np.linalg.norm(p - touch0[s])))
            else:
                touch0[s] = None
        if t > 0.05:
            FS.append(np.abs(d.actuator_force) / m.actuator_forcerange[:, 1])
        if t >= t_full and fall_t is None:
            supp.append(int(foot_on["R"]) + int(foot_on["L"]))
            for sd in "RL":
                if not foot_on[sd]:
                    h = min(d.geom_xpos[i][2] - m.geom_size[i][1] for i in pads[sd])
                    clear[sd] = max(clear[sd], h)
            yaw.append(math.degrees(math.atan2(R3[1, 0], R3[0, 0])))
        if k % 20 == 0:
            track.append((t, *(d.subtree_com[tb] * 1e3), tilt, int(foot_on["R"]), int(foot_on["L"])))
        if ren is not None and (k % every == 0 or any(abs(t - fa) < dt / 2 for fa in frames_at)):
            cam.lookat[:] = [d.subtree_com[tb][0], d.subtree_com[tb][1], 0.22]; cam.distance = 0.8
            cam.azimuth = 225; cam.elevation = -10
            ren.update_scene(d, cam); img = ren.render().copy()
            if video and k % every == 0 and t <= 10.0:
                frames.append(img)
            for fa in frames_at:
                if abs(t - fa) < dt / 2:
                    shots[fa] = img
        if fall_t is not None and t > fall_t + 1.0:
            break
    if video and frames:
        import imageio.v2 as iio
        iio.mimsave(video, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    t_walk0 = T_STAND + T_CROUCH
    steps_done = None if fall_t is None else (fall_t - t_walk0) / T
    res = {"kp_scale": kp_scale, "fallen": fall_t is not None, "fall_time_s": fall_t,
           "fall_in_step": fall_step, "steps_completed_before_fall": steps_done,
           "full_steps_survived": (N_FULL if fall_t is None else max(0.0, (fall_t - t_walk0) / T - N_RAMP)),
           "max_slip_mm": {s: v * 1e3 for s, v in slip.items()},
           "foot_foot_contacts": len(ff_contacts), "first_foot_foot": ff_contacts[:3],
           "first_nonpad_floor_contact": first_floor_nonpad,
           "force_over_stall": {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): float(fmax[i] / m.actuator_forcerange[i][1]) for i in range(m.nu)},
           "nod_crank_force_max": crank_f,
           "com_x_progress_mm": track[-1][1] - track[0][1], "final_com_mm": track[-1][1:4]}
    FS = np.array(FS); supp = np.array(supp)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    res["force_over_stall_p99"] = {nm_: float(np.percentile(FS[:, i], 99)) for i, nm_ in enumerate(names)}
    res["saturated_time_frac"] = {nm_: float(np.mean(FS[:, i] >= 0.999)) for i, nm_ in enumerate(names)}
    res["full_steps_support"] = {"single_support_frac": float(np.mean(supp == 1)), "double_support_frac": float(np.mean(supp == 2)),
                                 "no_support_frac": float(np.mean(supp == 0))}
    res["max_swing_foot_clearance_mm"] = {sd: v * 1e3 for sd, v in clear.items()}
    res["torso_yaw_deg_full_steps"] = [float(min(yaw)), float(max(yaw))] if yaw else None
    nominal = CASE["step"] * (N_FULL + 1)
    res["com_x_progress_vs_nominal"] = {"nominal_mm": nominal, "ratio": float(res["com_x_progress_mm"] / nominal)}
    return res, np.array(track), shots


out = {"case": {k: (float(np.degrees(v)) if k == "psi_amp" else v) for k, v in CASE.items()},
       "timeline": dict(stand=T_STAND, crouch=T_CROUCH, ramp_steps=N_RAMP, full_steps=N_FULL)}
FR = [T_STAND + T_CROUCH - 0.01, T_STAND + T_CROUCH + 1.5 * T, T_STAND + T_CROUCH + 3.5 * T, T_STAND + T_CROUCH + 6.5 * T]
r1, tr1, shots1 = run(1.0, None, FR)
out["kp_x1"] = r1
print("kp x1", {k: v for k, v in r1.items() if k not in ("force_over_stall", "saturated_time_frac", "force_over_stall_p99")})
print("   p99", {k: round(v, 2) for k, v in r1["force_over_stall_p99"].items()}, "sat", {k: round(v, 3) for k, v in r1["saturated_time_frac"].items() if v})
print("   force/stall", {k: round(v, 2) for k, v in r1["force_over_stall"].items()})
r4, tr4, shots4 = run(4.0, os.path.join(RES, "test3_openloop_walk_kp4.mp4"), FR)
out["kp_x4"] = r4
print("kp x4", {k: v for k, v in r4.items() if k not in ("force_over_stall", "saturated_time_frac", "force_over_stall_p99")})
print("   p99", {k: round(v, 2) for k, v in r4["force_over_stall_p99"].items()}, "sat", {k: round(v, 3) for k, v in r4["saturated_time_frac"].items() if v})
print("   force/stall", {k: round(v, 2) for k, v in r4["force_over_stall"].items()})
jdump(out, "test3_openloop_walk.json")

plt = korean_font()
fig = plt.figure(figsize=(15, 8.5))
for i, fa in enumerate(FR):
    for row, (shots, lab) in enumerate(((shots1, "기본 이득"), (shots4, "이득 ×4"))):
        a = fig.add_subplot(3, 4, 1 + i + 4 * row)
        if fa in shots:
            a.imshow(shots[fa])
        a.axis("off"); a.set_title(f"{lab} · {fa:.2f} s", fontsize=9)
a = fig.add_subplot(3, 1, 3)
for tr, lab, c in ((tr1, "기본 이득", "#2a6fdb"), (tr4, "이득 ×4", "#e07b39")):
    a.plot(tr[:, 0], tr[:, 4], color=c, label=f"몸통 기울기 ({lab})")
a.axvline(T_STAND + T_CROUCH, color="gray", ls=":"); a.axvline(T_STAND + T_CROUCH + N_RAMP * T, color="gray", ls=":")
a.set_xlabel("시간 (s) — 점선: 걸음 시작 · 진폭 키우기 끝"); a.set_ylabel("몸통 기울기 (°)"); a.grid(alpha=0.3); a.legend(fontsize=9)
def verdict(r):
    v = f"{r['full_steps_survived']:.1f}걸음 뒤 넘어짐" if r["fallen"] else "10걸음 버팀"
    return v + f"(발 들림 최대 {max(r['max_swing_foot_clearance_mm'].values()):.0f} mm, 한 발 지지 {100 * r['full_steps_support']['single_support_frac']:.0f}%)"
fig.suptitle(f"servo_study 보통 걸음 궤적을 되먹임 없이 재생하면 넘어지지는 않지만, 기본 이득은 발을 거의 못 들고 끈다\n"
             f"기본 이득: {verdict(r1)} · 이득 4배: {verdict(r4)} (진폭 다 키운 뒤 10걸음, 시뮬 실측)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(RES, "test3_openloop_walk.png"), dpi=100)
print("saved")
