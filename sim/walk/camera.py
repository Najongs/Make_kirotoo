"""머리 카메라(Pi Camera v2, 왼쪽 눈 렌즈) 영상: 걸으면서 머리 끄덕임(2b) 중 30 fps 640x480 mp4 + 영상이 쓸 만한지 수치.

python camera.py  → results/camera_eye_nod1Hz.mp4 (10 s), results/camera_side_nod1Hz.mp4, results/camera.json, results/camera_frames.png
수치: 카메라 각속도(광축 둘레 = 영상 회전, 좌우 · 상하 = 영상 흔들림) RMS · 최대, 한 프레임(1/30 s) 동안 화면 이동 픽셀,
      화면 아래 가장자리 광선이 바닥에 닿는 거리(그보다 가까운 바닥은 안 보임), 로봇 자기 몸이 화면을 가리는 비율(분할 렌더).
"""
import math, os, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from collections import deque
import numpy as np
import mujoco
from kw_common import build_model, scale_kp, Idx, sense, RES, CAM_FOVY, korean_font, jdump
from kw_ctrl import Controller, Footsteps, make_steps, rpy_of
from run_tests import BASE

W, H, FPS = 640, 480, 30
FPX = (H / 2) / math.tan(math.radians(CAM_FOVY / 2))      # 세로 화각 기준 초점거리(픽셀)


def landmarks(spec):
    """영상 판독용 표지물(충돌 없음): 바닥 위 0.2 m 간격 색 띠(0.2–1.0 m), 상자 셋, 2 m 앞 벽"""
    wb = spec.worldbody
    for i, x in enumerate((0.2, 0.4, 0.6, 0.8, 1.0)):
        g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.01, 0.3, 0.0005]; g.pos = [x, 0, 0.0005]
        g.rgba = [[0.9, 0.1, 0.1, 1], [0.95, 0.6, 0.1, 1], [0.9, 0.9, 0.1, 1], [0.2, 0.8, 0.2, 1], [0.2, 0.4, 0.95, 1]][i]
        g.contype = 0; g.conaffinity = 0
    for x, y, rgba in ((0.7, 0.15, [0.9, 0.2, 0.2, 1]), (1.1, -0.25, [0.2, 0.7, 0.2, 1]), (1.5, 0.3, [0.2, 0.3, 0.9, 1])):
        g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.05, 0.05, 0.05]; g.pos = [x, y, 0.05]; g.rgba = rgba
        g.contype = 0; g.conaffinity = 0
    g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.01, 1.5, 0.6]; g.pos = [2.2, 0, 0.6]; g.rgba = [0.95, 0.9, 0.7, 1]
    g.contype = 0; g.conaffinity = 0


def run_cam(stride=0.03, nod=1.0, T_rec=(3.0, 13.0), video=True, tag="nod1Hz"):
    c = dict(BASE)
    m, _ = build_model(camera=True, extra_bodies=landmarks); scale_kp(m, c["kp_scale"])
    mp, _ = build_model(); mp.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m); d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
    ix = Idx(m); kp = m.actuator_gainprm[:, 0].copy()
    dtc = 1.0 / c["ctrl_hz"]; nsub = int(round(dtc / m.opt.timestep))
    plan = Footsteps(make_steps(c["n_steps"], stride), 1.5, c["Ts"], c["Td"], 0.8, 0.8, (-0.007, 0.0))
    lo, hi = math.radians(-10), math.radians(20); t0 = 3.0
    head_fn = (lambda t: (min(1.0, t - t0) * (0.5 * (lo + hi) - 0.5 * (hi - lo) * math.cos(2 * math.pi * nod * (t - t0)))) if t > t0 else 0.0) if nod else None
    ctl = Controller(mp, kp, dtc, plan, crouch=c["crouch"], lift=c["lift"], k_dcm=c["k_dcm"], head_fn=head_fn)
    nd = int(round(c["delay_ms"] / 1000 / dtc)); buf = deque([d.ctrl.copy()] * (nd + 1), maxlen=nd + 1)
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    ren = mujoco.Renderer(m, H, W) if video else None
    side = mujoco.MjvCamera(); side.type = mujoco.mjtCamera.mjCAMERA_FREE
    every = int(round(1.0 / (FPS * m.opt.timestep)))
    frames, sides, poses, wcam, seg, shots = [], [], [], [], [], {}
    robot_geoms = np.where(m.geom_bodyid != 0)[0]          # 로봇 몸(world 에 붙은 바닥 · 표지물 제외)
    T_end = T_rec[1] + 0.01
    fallen = None
    for k in range(int(T_end / m.opt.timestep)):
        t = k * m.opt.timestep
        if k % nsub == 0:
            buf.append(ctl.step(t, sense(m, d, ix))); d.ctrl[:] = buf[0]
        mujoco.mj_step(m, d)
        if fallen is None and d.xmat[ix.torso][8] < math.cos(math.radians(35)):
            fallen = t
        if T_rec[0] <= t and k % every == 0 and len(poses) < int(round((T_rec[1] - T_rec[0]) * FPS)):   # 10 s = 300 프레임
            R = d.cam_xmat[cam_id].reshape(3, 3).copy(); p = d.cam_xpos[cam_id].copy()
            w = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_CAMERA, cam_id, w, 1)   # 카메라 좌표 각속도
            poses.append((t, p, R, math.degrees(d.qpos[ix.qadr["head_nod"]]), d.subtree_com[ix.torso].copy(),
                          max(d.xpos[ix.foot[s]][0] for s in "RL")))
            wcam.append(w[:3].copy())
            if ren is not None:
                ren.update_scene(d, camera="eye"); img = ren.render().copy(); frames.append(img)
                side.lookat[:] = [d.subtree_com[ix.torso][0] + 0.25, 0, 0.2]; side.distance = 1.4; side.azimuth = 200; side.elevation = -18
                ren.update_scene(d, side); sides.append(ren.render().copy())
                if len(frames) % 3 == 1:
                    ren.enable_segmentation_rendering(); ren.update_scene(d, camera="eye"); sg = ren.render().copy()
                    ren.disable_segmentation_rendering()
                    ids, typ = sg[..., 0], sg[..., 1]
                    own = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids, robot_geoms)
                    seg.append((t, float(own.mean()), float(own[H // 2:].mean())))
                hd = math.degrees(d.qpos[ix.qadr["head_nod"]])
                for key in (-10, 5, 20):
                    if key not in shots and abs(hd - key) < 1.0 and t > 4.5:
                        shots[key] = (img, t, hd)
    if ren is not None and frames:
        import imageio.v2 as iio
        iio.mimsave(os.path.join(RES, f"camera_eye_{tag}.mp4"), frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
        iio.mimsave(os.path.join(RES, f"camera_side_{tag}.mp4"), sides, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    # ---- 수치
    wc = np.degrees(np.array(wcam))           # 카메라 좌표: x 영상 오른쪽, y 영상 위, z 뒤(광축 반대)
    w_roll = wc[:, 2]; w_pan = wc[:, 1]; w_tilt = wc[:, 0]
    wn = np.linalg.norm(wc, axis=1)
    near = []; ctr = []; roll_img = []
    for (t, p, R, hd, com, toe) in poses:
        dn = R @ np.array([0, -math.tan(math.radians(CAM_FOVY / 2)), -1.0])   # 화면 아래 가장자리 가운데 광선
        dc = R @ np.array([0, 0, -1.0])
        def hit(dv):
            return (p + dv * (-p[2] / dv[2])) if dv[2] < -1e-6 else None
        hn = hit(dn); hc = hit(dc)
        fwd = np.array([dc[0], dc[1]]); fwd /= max(np.linalg.norm(fwd), 1e-9)
        near.append((hd, (hn[:2] - p[:2]) @ fwd if hn is not None else np.inf, (hn[0] - toe) if hn is not None else np.inf))
        ctr.append((hc[:2] - p[:2]) @ fwd if hc is not None else np.inf)
        # 영상 기울기: 카메라 x 축(영상 오른쪽)의 수평에 대한 각
        roll_img.append(math.degrees(math.asin(np.clip(R[2, 0], -1, 1))))
    near = np.array(near); roll_img = np.array(roll_img)
    def stat(a):
        return {"rms": float(np.sqrt(np.mean(a ** 2))), "max": float(np.abs(a).max())}
    out = {"stride_m": stride, "nod_hz": nod, "record_s": [poses[0][0], poses[-1][0]] if poses else None, "fallen": fallen,
           "cam_height_m": [float(min(p[1][2] for p in poses)), float(max(p[1][2] for p in poses))],
           "omega_total_dps": stat(wn), "omega_optical_axis_dps(영상 회전)": stat(w_roll),
           "omega_pan_dps(좌우)": stat(w_pan), "omega_tilt_dps(상하)": stat(w_tilt),
           "px_per_frame_at_30fps": {"rms": float(np.sqrt(np.mean(wn ** 2))) * math.pi / 180 * FPX / FPS,
                                     "max": float(wn.max()) * math.pi / 180 * FPX / FPS},
           "focal_px": FPX,
           "image_roll_deg": [float(roll_img.min()), float(roll_img.max())],
           "near_ground_visible_from_camera_m": {"min": float(near[:, 1].min()), "max": float(near[:, 1][np.isfinite(near[:, 1])].max()),
                                                  "median": float(np.median(near[:, 1]))},
           "near_ground_ahead_of_toe_m": {"min": float(near[:, 2].min()), "median": float(np.median(near[:, 2]))},
           "head_deg_range": [float(near[:, 0].min()), float(near[:, 0].max())],
           "self_occlusion_frac": {"mean": float(np.mean([s[1] for s in seg])) if seg else None,
                                   "max": float(np.max([s[1] for s in seg])) if seg else None}}
    # 머리각별 아래 가장자리 거리(5° 칸)
    bins = {}
    for hd, dn_, _ in near:
        b = int(round(hd / 5.0) * 5)
        bins.setdefault(b, []).append(dn_)
    out["near_ground_by_head_deg"] = {str(b): float(np.median(v)) for b, v in sorted(bins.items())}
    return out, shots, (np.array([p[0] for p in poses]), wc, near, roll_img)


if __name__ == "__main__":
    res = {}
    o1, shots, ser = run_cam(0.03, 1.0, video=True, tag="nod1Hz")
    res["walk_0.046mps_nod_1Hz"] = o1
    o2, _, ser2 = run_cam(0.03, 0.5, video=False)
    res["walk_0.046mps_nod_0.5Hz"] = o2
    o3, _, ser3 = run_cam(0.03, None, video=False)
    res["walk_0.046mps_no_nod"] = o3
    o4, _, _ = run_cam(0.0, None, video=False)
    res["step_in_place_no_nod"] = o4
    jdump(res, os.path.join(RES, "camera.json"))
    for k, v in res.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk in ("omega_total_dps", "px_per_frame_at_30fps", "image_roll_deg",
                                                                       "near_ground_visible_from_camera_m", "self_occlusion_frac", "fallen")}, ensure_ascii=False))
    # 그림: 머리각 −10 / +5 / +20 의 눈 영상 + 각속도 시계열
    plt = korean_font()
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    for i, key in enumerate((-10, 5, 20)):
        a = fig.add_subplot(gs[0, i])
        if key in shots:
            img, t, hd = shots[key]; a.imshow(img); a.set_title(f"머리 {hd:+.0f}° (+ 아래) · {t:.2f} s", fontsize=10)
        a.axis("off")
    a = fig.add_subplot(gs[1, :])
    t, wc, near, rimg = ser
    a.plot(t, wc[:, 2], label="광축 둘레(영상 회전)", lw=1); a.plot(t, wc[:, 1], label="좌우(팬)", lw=1); a.plot(t, wc[:, 0], label="상하(틸트)", lw=1)
    a.plot(ser3[0], np.linalg.norm(ser3[1], axis=1), label="끄덕임 없이 걸을 때 전체 크기", color="gray", lw=1, ls="--")
    a.set_xlabel("시간 (s)"); a.set_ylabel("카메라 각속도 (°/s)"); a.grid(alpha=.3); a.legend(fontsize=9, ncol=4)
    fig.suptitle(f"걸으며 1 Hz 로 끄덕이면 카메라 각속도 RMS {o1['omega_total_dps']['rms']:.0f}°/s · 최대 {o1['omega_total_dps']['max']:.0f}°/s, "
                 f"영상은 좌우로 {o1['image_roll_deg'][0]:.0f}~{o1['image_roll_deg'][1]:+.0f}° 기운다\n"
                 f"바닥은 카메라 앞 {o1['near_ground_visible_from_camera_m']['min']:.2f} m 보다 가까이는 안 보인다 "
                 f"(보통 걸음 0.046 m/s, 기본 제어기, 시뮬 실측)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(RES, "camera_frames.png"), dpi=90)
    print("saved")
