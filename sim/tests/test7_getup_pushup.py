"""Test 7 (extra): lying face down, both arms swing slowly from 0 to -120 deg (hands come under the chest and
push it up). Quasi-static (6 s ramp). Shoulder actuator force vs the handover estimate 1.0-1.1 N*m (XC430 1.75).
Legs hold the zero pose. Run: cd sim && uv run python tests/test7_getup_pushup.py -> results/test7_getup_pushup.json, .png
"""
import math, numpy as np
from common import *

m, d = load(); mujoco.mj_resetDataKeyframe(m, d, 0)
R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)      # +90 deg about y: front (+x) faces the floor
d.qpos[3:7] = quat_of(R); d.qpos[0:3] = [0, 0, 0.0]
mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
# lift so that the lowest collision vertex is 3 mm above the floor
zmin = 1e9
for i in range(m.ngeom):
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[i]:
        mid = m.geom_dataid[i]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[a:a + n] @ d.xmat[0].reshape(3, 3).T if False else m.mesh_vert[a:a + n] @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        zmin = min(zmin, V[:, 2].min())
d.qpos[2] += 0.003 - zmin
tb = bid(m, "torso"); aR, aL = aid(m, "R_arm_swing"), aid(m, "L_arm_swing")
T_SETTLE, T_RAMP, T_HOLD = 1.0, 6.0, 1.0
rec = []; frames = {}
ren = mujoco.Renderer(m, 360, 480); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
for k in range(int((T_SETTLE + T_RAMP + T_HOLD) / m.opt.timestep)):
    t = k * m.opt.timestep
    a = -math.radians(120) * min(max((t - T_SETTLE) / T_RAMP, 0.0), 1.0)
    d.ctrl[:] = 0; d.ctrl[aR] = a; d.ctrl[aL] = a
    mujoco.mj_step(m, d)
    if k % 20 == 0:
        x_body = d.xmat[tb].reshape(3, 3)[:, 0]            # body front axis in world
        pitch_up = math.degrees(math.asin(np.clip(x_body[2], -1, 1))) + 90   # 0 = lying face down, 90 = upright
        rec.append((t, math.degrees(a), math.degrees(d.qpos[qa(m, "R_arm_swing")]), d.actuator_force[aR], d.actuator_force[aL],
                    d.subtree_com[tb][2] * 1e3, pitch_up))
    for tf in (T_SETTLE, T_SETTLE + T_RAMP * 0.5, T_SETTLE + T_RAMP * 0.75, T_SETTLE + T_RAMP + T_HOLD - 0.01):
        if abs(t - tf) < m.opt.timestep / 2:
            cam.lookat[:] = [d.subtree_com[tb][0], d.subtree_com[tb][1], 0.08]; cam.distance = 0.9; cam.azimuth = 270; cam.elevation = -10
            ren.update_scene(d, cam); frames[round(tf, 2)] = ren.render().copy()
rec = np.array(rec)
ramp = rec[(rec[:, 0] > T_SETTLE + 0.1)]
fR, fL = np.abs(ramp[:, 3]), np.abs(ramp[:, 4])
out = {"peak_shoulder_force_Nm": {"R": float(fR.max()), "L": float(fL.max())},
       "peak_over_stall_XC430_1.75": float(max(fR.max(), fL.max()) / 1.75),
       "peak_over_stall_emanual_1.7": float(max(fR.max(), fL.max()) / 1.7),
       "arm_angle_at_peak_deg": float(ramp[np.argmax(fR), 2]),
       "arm_reached_deg": float(rec[-1, 2]), "arm_target_deg": -120.0,
       "torso_pitch_up_final_deg": float(rec[-1, 6]), "com_height_mm": [float(rec[:, 5].min()), float(rec[:, 5].max())],
       "saturated_frac_during_ramp": float(np.mean(np.maximum(fR, fL) >= 1.75 * 0.999)),
       "handover_estimate_Nm": [1.0, 1.1]}
jdump(out, "test7_getup_pushup.json"); print(out)
plt = korean_font()
fig = plt.figure(figsize=(14, 7))
for i, (tf, img) in enumerate(sorted(frames.items())):
    a = fig.add_subplot(2, 4, 1 + i); a.imshow(img); a.axis("off"); a.set_title(f"{tf:.2f} s", fontsize=9)
a = fig.add_subplot(2, 1, 2)
a.plot(rec[:, 0], np.abs(rec[:, 3]), label="오른 어깨 |힘| (N·m)", color="#2a6fdb")
a.plot(rec[:, 0], np.abs(rec[:, 4]), label="왼 어깨 |힘| (N·m)", color="#7fb3ff", ls="--")
a.axhline(1.75, color="#c0392b", ls="--", label="XC430 정지토크 1.75"); a.axhspan(1.0, 1.1, color="#f5c26b", alpha=0.4, label="인계 문서 추정 1.0–1.1")
a2 = a.twinx(); a2.plot(rec[:, 0], rec[:, 6], color="#555", lw=1, label="상체 들림 (°)"); a2.set_ylabel("상체 들림 (°, 0 = 엎드림)")
a.set_xlabel("시간 (s)"); a.set_ylabel("어깨 토크 (N·m)"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
fig.suptitle(f"엎드려 팔로 상체를 밀 때 어깨 최대 토크 {max(fR.max(), fL.max()):.2f} N·m (인계 추정 1.0–1.1, 정지토크 대비 {out['peak_over_stall_XC430_1.75']:.2f}), "
             f"상체 {out['torso_pitch_up_final_deg']:.0f}° 들림 — 준정적, 시뮬 실측", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(RES, "test7_getup_pushup.png"), dpi=100)
print("saved")
