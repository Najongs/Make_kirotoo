"""질문 4: 엎드림 · 누움에서 일어나기 — 간단한 키프레임(관절 목표를 시간에 따라 선형으로 옮김), 위치 이득 2배, 서보 정지토크 · 속도 한계 안.
시뮬에서만 무릎 · 고관절 피치 범위를 넓혀(관절 range 와 액추에이터 ctrlrange 둘 다) 「무릎을 최대 몇 도 굽혀야 일어나는가」를 잰다.

엎드림(앞이 바닥) 순서(관절 각은 좌우 같음, 부호: 고관절 피치 − 가 굽힘, 무릎 + 가 굽힘, 발목 피치 − 가 발등 굽힘, 팔 − 가 앞으로):
  0) 0.8 s 영점에서 가라앉음
  1) 팔을 앞으로(바닥 쪽) 돌려 손으로 가슴을 받침 + 다리를 접음(고관절 −H, 무릎 K, 발목 −A) → 무릎 · 손으로 엎드린 자세
  2) 팔을 더 돌려(머리 위 쪽) 몸통을 뒤로 밀어 무게중심을 발 위로 → 발로 쪼그려 앉은 자세
  3) 다리를 천천히 폄 → 섬
성공: 끝 1 s 동안 몸통 기울기 < 15° 이고 무게중심 높이가 서 있는 높이의 90 % 이상.
누움(등이 바닥): 1축 팔로 상체를 세우기는 어려워, 「옆으로 굴러 엎드림이 된 뒤 위 순서」 대신 팔 · 다리로 앉아 일어나기를 한 번 시도만 한다.

usage: cd sim/walk/speed && ../../.venv/bin/python getup.py → results/getup.json, results/getup_frames_*.png"""
import sys, os, math, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import mujoco
from sp_common import CUR, hrA, Idx2
from kw_common import build_model, scale_kp, jdump, korean_font


def widen(knee_max, hip_min):
    def f(spec):
        for s in "RL":
            j = spec.joint(f"{s}_knee"); j.range = [0.0, knee_max]
            j = spec.joint(f"{s}_hip_pitch"); j.range = [hip_min, j.range[1]]
            for a in spec.actuators:
                if a.name == f"{s}_knee": a.ctrlrange = [0.0, knee_max]
                if a.name == f"{s}_hip_pitch": a.ctrlrange = [hip_min, a.ctrlrange[1]]
    return f


def place(m, d, R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(9)); d.qpos[3:7] = q; d.qpos[0:3] = 0
    mujoco.mj_kinematics(m, d)
    zmin = 1e9
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[i]:
            mid = m.geom_dataid[i]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V = m.mesh_vert[a:a + n] @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
            zmin = min(zmin, V[:, 2].min())
    d.qpos[2] += 0.003 - zmin


def run(keys, start="prone", servo_map=CUR, knee_max=1.745, hip_min=-1.396, frames_at=(), T_hold=1.5, L=0.0405):
    m, _ = build_model(hip_roll=hrA(L), servo_map=servo_map, extra_bodies=widen(knee_max, hip_min))
    scale_kp(m, 2.0); ix = Idx2(m)
    d = mujoco.MjData(m); d.qpos[:] = m.qpos0
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float) if start == "prone" else np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    place(m, d, R)
    d0 = mujoco.MjData(m); d0.qpos[:] = m.qpos0; mujoco.mj_forward(m, d0); z_stand = float(d0.subtree_com[1][2])
    J = {"hip": "hip_pitch", "knee": "knee", "ankle": "ankle_pitch", "arm": "arm_swing"}
    cur = {k: 0.0 for k in J}; sched = [(0.8, dict(cur))]
    for dur, tg in keys:
        c = dict(sched[-1][1]); c.update(tg); sched.append((dur, c))
    T = sum(s[0] for s in sched) + T_hold
    ren = None; shots = {}
    if frames_at:
        ren = mujoco.Renderer(m, 300, 400); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    tq = np.zeros(m.nu); sp = np.zeros(m.nu); hist = []
    n = int(T / m.opt.timestep); tb = 1
    for k in range(n):
        t = k * m.opt.timestep
        # 지금 구간에서 선형 보간
        t0 = 0.0; prev = sched[0][1]; tgt = sched[-1][1]
        for i, (dur, c) in enumerate(sched):
            if t < t0 + dur:
                p = sched[i - 1][1] if i else c
                s = (t - t0) / dur; tgt = {kk: p[kk] + (c[kk] - p[kk]) * s for kk in c}; break
            t0 += dur
        for a, nm in enumerate(ix.act):
            for kk, jn in J.items():
                if nm.endswith(jn):
                    d.ctrl[a] = tgt[kk]
        d.ctrl[:] = np.clip(d.ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
        if t > 0.8:
            w = np.array([d.qvel[ix.vadr[j]] for j in ix.act_j]); b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
            tq = np.maximum(tq, np.abs(d.actuator_force - b * w) / ix.stall); sp = np.maximum(sp, np.abs(w) / ix.w0)
        if k % 20 == 0:
            hist.append((t, d.subtree_com[1][2], math.degrees(math.acos(np.clip(d.xmat[tb][8], -1, 1)))))
        for tf in frames_at:
            if abs(t - tf) < m.opt.timestep / 2 and ren is not None:
                cam.lookat[:] = [d.subtree_com[1][0], d.subtree_com[1][1], 0.12]; cam.distance = 0.8; cam.azimuth = 270; cam.elevation = -5
                ren.update_scene(d, cam); shots[tf] = ren.render().copy()
    H = np.array(hist); last = H[:, 0] > T - 1.0
    ok = bool((H[last, 2] < 15).all() and (H[last, 1] > 0.9 * z_stand).all())
    names = [nm for nm in ix.act if nm.startswith("R_")]
    mid = H[:, 0] > 2.0
    res = dict(ok=ok, best_tilt=float(H[mid, 2].min()), best_z=float(H[mid, 1].max()), z_end=float(H[-1, 1]), z_stand=z_stand, tilt_end=float(H[-1, 2]), T=T,
               torque_over_stall={nm[2:]: float(tq[ix.act.index(nm)]) for nm in names},
               speed_over_w0={nm[2:]: float(sp[ix.act.index(nm)]) for nm in names},
               knee_used_deg=math.degrees(max(c["knee"] for _, c in sched)), hip_used_deg=math.degrees(min(c["hip"] for _, c in sched)))
    return res, shots, H


def prone_keys(K, H, A=-0.87, arm1=-1.6, arm2=-2.09, lean=0.0):
    """K 무릎 최대, H 고관절 최대 굽힘(음수), A 발목. 마지막 펴기: 쪼그림 → 섬"""
    return [(1.0, dict(arm=arm1)),
            (1.2, dict(hip=H, knee=K, ankle=A)),
            (1.2, dict(arm=arm2)),
            (0.8, dict(hip=H * 0.6 + lean, arm=arm2)),
            (2.0, dict(hip=0.0, knee=0.0, ankle=0.0, arm=0.0))]


def prone_keys2(K, H1, A, arm2=-2.09):
    """엎드림: 팔로 가슴 받침 → 다리 접기(고관절 H1, 무릎 K, 발목 A) → 팔을 머리 위 쪽으로 더 돌려 몸통을 뒤로 → 발목 평평 · 고관절 조금 폄 → 섬"""
    return [(0.8, dict(arm=-1.57)), (1.0, dict(hip=H1, knee=K, ankle=A)), (1.0, dict(arm=arm2)),
            (0.8, dict(ankle=-0.87, hip=H1 + 0.3)), (1.0, dict(hip=-K / 2 - 0.3, ankle=-0.87)), (2.0, dict(hip=0.0, knee=0.0, ankle=0.0, arm=0.0))]


def supine_keys(K, H1, A):
    """누움: 팔을 뒤로(+) 돌려 상체를 밀어 앉음 → 다리 접기 → 몸을 앞으로 숙이며 팔을 앞으로 → 섬"""
    return [(0.8, dict(arm=2.09)), (1.0, dict(hip=H1, knee=K, ankle=A)), (1.0, dict(arm=0.5, hip=H1 - 0.3)),
            (1.0, dict(arm=-1.0, ankle=-0.87)), (2.0, dict(hip=0.0, knee=0.0, ankle=0.0, arm=0.0))]


def prone_throw(K, H1, tp):
    """반동: 다리를 먼저 깊이 접어(무릎 · 고관절) 엉덩이를 발 위로 가져간 뒤, 팔을 tp 초 만에 −30° → −120° 로 휘둘러 몸통을 뒤로 던지고
    곧바로 쪼그림(무게중심이 발 위인 자세) → 섬"""
    return [(0.8, dict(arm=-0.5)), (1.0, dict(hip=H1, knee=K, ankle=-0.87)), (0.6, dict(arm=-1.2)),
            (tp, dict(arm=-2.09, hip=-K * 0.7)), (0.6, dict(arm=-1.0)), (2.0, dict(hip=0.0, knee=0.0, ankle=0.0, arm=0.0))]


ARMS = {"XL430-W250 (지금)": "XL430-W250", "XC430-W240": "XC430-W240", "XM430-W350": "XM430-W350"}


def job(a):
    start, K, H1, A, armn = a
    keys = prone_throw(K, H1, A) if start == "throw" else (prone_keys2(K, H1, A) if start == "prone" else supine_keys(K, H1, A))
    try:
        r, _, _ = run(keys, start=start, servo_map=dict(CUR, arm_swing=ARMS[armn]), knee_max=max(1.745, K), hip_min=min(-1.396, H1 - 0.35))
    except Exception as e:
        r = dict(ok=False, error=repr(e))
    return dict(start=start, K_deg=math.degrees(K), H1_deg=math.degrees(H1), A=A, arm=armn, **r)


if __name__ == "__main__":
    from multiprocessing import Pool
    if len(sys.argv) > 1 and sys.argv[1] == "throw":
        args = [("throw", K, H1, tp, an) for K in map(math.radians, (120, 135, 150)) for H1 in (-1.8, -2.2) for tp in (0.15, 0.3, 0.5) for an in ARMS]
        with Pool(min(60, len(args))) as P:
            rows = P.map(job, args)
        jdump(rows, os.path.join(HERE, "results", "getup_throw.json"))
        for r in rows:
            print(r["start"], round(r["K_deg"]), round(r["H1_deg"]), r["A"], r["arm"], r.get("ok"), "best tilt", round(r.get("best_tilt", 0)), "best z", round(r.get("best_z", 0) * 1e3),
                  {k: round(v, 2) for k, v in r.get("torque_over_stall", {}).items() if k in ("knee", "hip_pitch", "arm_swing", "ankle_pitch")})
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        args = [(st, K, H1, A, an) for st in ("prone", "supine") for K in map(math.radians, (100, 120, 135, 150))
                for H1 in (-1.4, -1.8, -2.2) for A in (-0.87, 0.38) for an in ARMS]
        with Pool(min(72, len(args))) as P:
            rows = P.map(job, args)
        jdump(rows, os.path.join(HERE, "results", "getup.json"))
        for r in rows:
            print(r["start"], round(r["K_deg"]), round(r["H1_deg"]), r["A"], r["arm"], r.get("ok"), round(r.get("z_end", 0) * 1e3), round(r.get("tilt_end", 0)),
                  {k: round(v, 2) for k, v in r.get("torque_over_stall", {}).items() if k in ("knee", "hip_pitch", "arm_swing", "ankle_pitch")})
        sys.exit()
    import itertools
    rows = []
    r, shots, H = run(prone_keys(2.4, -1.9), knee_max=2.6, hip_min=-2.2, frames_at=(0.8, 1.8, 3.0, 4.2, 5.0, 6.0, 7.0))
    print(r)
    plt = korean_font()
    fig, ax = plt.subplots(1, len(shots), figsize=(3 * len(shots), 3))
    for a, (tf, img) in zip(ax, sorted(shots.items())):
        a.imshow(img); a.axis("off"); a.set_title(f"{tf:.1f} s")
    fig.savefig(os.path.join(HERE, "results", "getup_try.png"), dpi=80)
