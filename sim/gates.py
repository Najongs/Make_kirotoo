"""Gates A-D for sim/model/kitoru_v17.xml against the design code (code_v17).

Run:  cd sim && MUJOCO_EGL_DEVICE_ID=2 uv run python gates.py
Out:  gates/gate_log.txt, gates/gate_B_compare.png
"""
import json, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code_v17")
sys.path.insert(0, CODE)
import render17 as R17
r = R17.r
from scipy.spatial import ConvexHull

GD = os.path.join(HERE, "gates"); os.makedirs(GD, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s); sys.stdout.flush()

XML = os.path.join(HERE, "model", "kitoru_v17.xml")
XMLF = os.path.join(HERE, "model", "kitoru_v17_fixed.xml")
LOOP = json.load(open(os.path.join(HERE, "model", "closed_loop_fit.json")))
MASS = json.load(open(os.path.join(CODE, "mass_build17_light.json")))
MERGE = {"nod_rod": "torso", "ear_rod": "head", "R_hand_flap": "R_arm", "L_hand_flap": "L_arm"}
links = r.links
PV = np.polynomial.polynomial.polyval
cH, cE = np.array(LOOP["polycoef_head"]), np.array(LOOP["polycoef_ear"])
A_TAB, B_TAB, P_TAB = (np.radians(LOOP[k]) for k in ("crank_deg", "beta_deg", "ear_deg"))

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)


def qadr(name):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]


def bid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def origin(ln):
    return np.array(links[ln]["origin"], float)


def zero():
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)


log("=== Limits actually compiled: forcelimited %d/13, ctrllimited %d/13, hinge range limited %d/15" % (
    int(m.actuator_forcelimited.sum()), int(m.actuator_ctrllimited.sum()), int(m.jnt_limited[1:].sum())))

# ------------------------------------------------------------------ A: mass
log("=== Gate A: total mass and centre of mass (zero pose)")
zero()
M = m.body_mass[1:].sum() * 1e3
C = d.subtree_com[bid("torso")] * 1e3
Mref, Cref = MASS["total_g"], np.array(MASS["com"])
okA = abs(M - Mref) <= 1.0 and np.linalg.norm(C - Cref) <= 0.5
log(f"  sim: {M:.3f} g, com ({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}) mm")
log(f"  design file: {Mref:.3f} g, com ({Cref[0]:.3f}, {Cref[1]:.3f}, {Cref[2]:.3f}) mm")
log(f"  diff: {M - Mref:+.2e} g, |dcom| {np.linalg.norm(C - Cref):.2e} mm -> {'PASS' if okA else 'FAIL'}")
log("  (merged nod_rod/ear_rod/hand_flaps keep mass and first moment exactly at the zero pose; they differ only when"
    " those links would move: nod_rod 1.0 g, ear_rod 1.7 g, flaps 5.3 g each)")

# ------------------------------------------------------------------ C: pads on the floor
log("=== Gate C: lowest foot pad point at the zero pose")
zero()
pads = [i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").count("_pad")]
zmin_pad = min(d.geom_xpos[i][2] - m.geom_size[i][1] for i in pads) * 1e3
zmax_pad = max(d.geom_xpos[i][2] - m.geom_size[i][1] for i in pads) * 1e3
zcol = []
for i in range(m.ngeom):
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[i]:
        mid = m.geom_dataid[i]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[a:a + n] @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        zcol.append((V[:, 2].min() * 1e3, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)))
zc = min(zcol)
okC = abs(zmin_pad) <= 0.5 and abs(zmax_pad) <= 0.5
log(f"  {len(pads)} pads, bottom z min {zmin_pad:.4f} mm, max {zmax_pad:.4f} mm -> {'PASS' if okC else 'FAIL'}")
log(f"  lowest non-pad collision hull: {zc[1]} at z {zc[0]:.3f} mm (should be > 0: only pads touch)")
rr = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), c.dist)
      for c in d.contact[:d.ncon]]
self_c = [x for x in rr if "floor" not in (x[0], x[1])]
log(f"  contacts at zero pose: {d.ncon} (robot-robot {len(self_c)}) {self_c[:6]}")

# ------------------------------------------------------------------ B: pose comparison
log("=== Gate B: non-zero pose, MuJoCo vs design code transforms (geom4.rot chain via render17.CHAINS)")
POSE = {"R_hip_yaw": 20.0, "R_knee": 60.0, "R_hip_pitch": -30.0, "L_ankle_roll": 15.0, "R_arm_swing": 90.0}
NP = R17.nod_pose(15.0)
crank = NP["nod_crank"]
head_poly = math.degrees(PV(math.radians(crank), cH)); ear_poly = math.degrees(PV(math.radians(crank), cE))
log(f"  nod_crank for head +15 deg: {crank:.4f} deg (brentq); poly head {head_poly:.4f}, ear {ear_poly:.4f} "
    f"(brentq ear {NP['ear_axle']:.4f})")
# design: rods / flaps follow their MuJoCo simplification (welded to the parent body)
r.CHAINS["ear_rod"] = ["head_nod"]; r.CHAINS["R_hand_flap"] = ["R_arm_swing"]; r.CHAINS["L_hand_flap"] = ["L_arm_swing"]
ANG = dict(POSE); ANG.update({"head_nod": 15.0, "nod_crank": crank, "ear_axle": NP["ear_axle"], "nod_rod": 0.0,
                              "R_hand_flap": 0.0, "L_hand_flap": 0.0})
mujoco.mj_resetDataKeyframe(m, d, 0)
for k, v in POSE.items():
    d.qpos[qadr(k)] = math.radians(v)
d.qpos[qadr("nod_crank")] = math.radians(crank)
d.qpos[qadr("head_nod")] = math.radians(head_poly)
d.qpos[qadr("ear_axle")] = math.radians(ear_poly)
mujoco.mj_forward(m, d)


def sim_pt(ln, p):
    b = bid(MERGE.get(ln, ln))
    return d.xpos[b] * 1e3 + d.xmat[b].reshape(3, 3) @ (np.asarray(p, float) - origin(MERGE.get(ln, ln)))


def des_pt(ln, p):
    M4 = r.link_pose(ln, ANG)
    return M4[:3, :3] @ np.asarray(p, float) + M4[:3, 3]


def solid_pts(ln, key=None):
    P = []
    for pn, ss in links[ln]["parts"].items():
        if key and key not in pn:
            continue
        for s in ss:
            T, _, caps = r.tris_of(s)
            if len(T):
                P.append(T.reshape(-1, 3))
    return np.concatenate(P)


checks = []
for ln in links:
    checks.append((f"com {ln}", ln, MASS["links"][ln]["com_world"]))
for S in "RL":
    for p in [(-62, -63), (-62, -17), (48, -63), (48, -17)]:
        checks.append((f"{S}_foot pad", f"{S}_foot", (p[0], p[1] if S == "R" else -p[1], 0.0)))
    P = solid_pts(f"{S}_foot"); checks.append((f"{S}_foot toe corner", f"{S}_foot", P[np.argmax(P[:, 0] - 0.01 * P[:, 2])]))
    P = solid_pts(f"{S}_arm", "hand"); checks.append((f"{S}_hand tip", f"{S}_arm", P[np.argmin(P[:, 2])]))
P = solid_pts("ears"); checks.append(("ear tip", "ears", P[np.argmax(P[:, 2])]))
P = solid_pts("head", "face"); checks.append(("face front", "head", P[np.argmax(P[:, 0])]))
P = solid_pts("nod_crank"); checks.append(("crank far point", "nod_crank", P[np.argmax(np.linalg.norm(P - origin("nod_crank"), axis=1))]))
worst = 0.0
for tag, ln, p in checks:
    a, b = sim_pt(ln, p), des_pt(ln, p); e = np.linalg.norm(a - b); worst = max(worst, e)
    if any(k in tag for k in ("pad", "tip", "toe", "face", "crank", "com R_foot", "com L_foot", "com R_arm", "com head", "com ears")):
        log(f"  {tag:22s} sim ({a[0]:8.2f},{a[1]:8.2f},{a[2]:8.2f})  design ({b[0]:8.2f},{b[1]:8.2f},{b[2]:8.2f})  diff {e:.2e} mm")
okB = worst < 0.1
log(f"  max position diff over {len(checks)} points (all link coms, pads, toes, hands, ear, face, crank): {worst:.3e} mm"
    f" -> {'PASS' if okB else 'FAIL'} (limit 0.1 mm; the residual comes from the 4th-order loop polynomial)")
# moved-distance sanity: the pose must actually move the checked points
moved = np.linalg.norm(des_pt("R_foot", (48, -63, 0)) - np.array([48, -63, 0.0]))
log(f"  (R_foot front pad moved {moved:.1f} mm from its zero-pose position)")

# renders: design (render6 rasteriser) vs MuJoCo, same view directions
from PIL import Image
VIEWS = [(15, -35), (0, 0), (0, -90), (20, 140)]
des_imgs = [r.render(e, a, ANG, scale=2.2) for e, a in VIEWS]
m.vis.global_.offwidth = 1600; m.vis.global_.offheight = 1600
ren = mujoco.Renderer(m, 900, 900)
opt = mujoco.MjvOption(); opt.geomgroup[:] = [0, 1, 0, 0, 0, 0]
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
ctr = np.array([0.0, 0.0, 0.25]); dist = 8.0
m.vis.global_.fovy = math.degrees(2 * math.atan(0.34 / dist))
mj_imgs = []
for e, a in VIEWS:
    cam.lookat[:] = ctr; cam.distance = dist; cam.azimuth = a + 180.0; cam.elevation = -e
    ren.update_scene(d, cam, opt); mj_imgs.append(ren.render().copy())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in (os.path.expanduser("~/.local/share/fonts/NotoSansKR.ttf"),):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
fig, ax = plt.subplots(2, len(VIEWS), figsize=(4.2 * len(VIEWS), 10.4), gridspec_kw=dict(hspace=0.18))
for i, (e, a) in enumerate(VIEWS):
    ax[0, i].imshow(des_imgs[i]); ax[0, i].set_title(f"설계 코드 렌더 (render6)\n고도 {e}°, 방위 {a}°", fontsize=10)
    ax[1, i].imshow(mj_imgs[i]); ax[1, i].set_title(f"MuJoCo 렌더 (kitoru_v17.xml)\n같은 시점", fontsize=10)
    for k in (0, 1):
        ax[k, i].axis("off")
fig.suptitle("같은 자세에서 설계 코드와 MuJoCo 모델의 모양이 일치한다 (오른 고관절 요 20°, 오른 고관절 피치 −30°, 오른 무릎 60°,\n"
             "왼 발목 롤 15°, 오른팔 90°, 머리 +15°(크랭크 %.2f°)) — 점 %d개 위치 차이 최대 %.1e mm, 시뮬 실측. "
             "막대·손 날개는 양쪽 모두 부모에 고정" % (crank, len(checks), worst), fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.04, hspace=0.18)
fig.savefig(os.path.join(GD, "gate_B_compare.png"), dpi=110)
plt.close(fig)
log("  saved gates/gate_B_compare.png")

# collision hulls at the zero pose (what touches the floor / the other leg)
zero()
opt2 = mujoco.MjvOption(); opt2.geomgroup[:] = [0, 0, 0, 1, 0, 0]
imgs = []
for e, a in ((15, -35), (0, -90)):
    cam.lookat[:] = ctr; cam.distance = dist; cam.azimuth = a + 180.0; cam.elevation = -e
    ren.update_scene(d, cam, opt2); imgs.append(ren.render().copy())
fig, ax = plt.subplots(1, 2, figsize=(9, 5))
for i in range(2):
    ax[i].imshow(imgs[i]); ax[i].axis("off")
fig.suptitle("충돌 형상: 부품마다 볼록 껍질 %d개 + 발 패드 원기둥 8개 (영점 자세, 주황 = 바닥과만, 분홍 = 바닥 + 다른 다리)"
             % sum(1 for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[i]), fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.03)
fig.savefig(os.path.join(GD, "collision_hulls.png"), dpi=110); plt.close(fig)
log("  saved gates/collision_hulls.png")

# leg-vs-leg bit (4): toe-in yaw must make the two feet touch, the torso must not
log("=== Collision bits: R_hip_yaw +30 deg (toe swings across the midline)")
mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[2] += 0.01; d.qpos[qadr("R_hip_yaw")] = math.radians(30); mujoco.mj_forward(m, d)
names = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
         for c in d.contact[:d.ncon]]
legleg = [x for x in names if "floor" not in x]
log(f"  robot-robot contacts: {len(legleg)} {sorted(set(legleg))[:6]}")

# ------------------------------------------------------------------ closed loop in simulation (fixed torso)
log("=== Closed loop in simulation (torso fixed): crank servo driven to its range ends and swept")
mf = mujoco.MjModel.from_xml_path(XMLF); df = mujoco.MjData(mf)
ja = {n: mf.jnt_qposadr[mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("head_nod", "nod_crank", "ear_axle")}
ac = mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_ACTUATOR, "nod_crank")
lo, hi = mf.actuator_ctrlrange[ac]
mujoco.mj_resetDataKeyframe(mf, df, 0)
viol_p, viol_t, viol_e = 0.0, 0.0, 0.0
T = 6.0; n = int(T / mf.opt.timestep)
for k in range(n):
    t = k * mf.opt.timestep
    df.ctrl[ac] = hi if t < 1.5 else lo if t < 3.0 else 0.5 * (hi + lo) + 0.5 * (hi - lo) * math.sin(2 * math.pi * 1.0 * (t - 3.0))
    mujoco.mj_step(mf, df)
    qa, qh, qe = df.qpos[ja["nod_crank"]], df.qpos[ja["head_nod"]], df.qpos[ja["ear_axle"]]
    viol_p = max(viol_p, abs(qh - PV(qa, cH)))
    if A_TAB[0] <= qa <= A_TAB[-1]:
        viol_t = max(viol_t, abs(qh - np.interp(qa, A_TAB, B_TAB)))
    viol_e = max(viol_e, abs(qe - PV(qa, cE)))
okL = math.degrees(max(viol_p, viol_e)) <= 0.5
log(f"  6 s: crank -> {math.degrees(hi):.2f} deg, -> {math.degrees(lo):.2f} deg, then 1 Hz full-range sine")
log(f"  max |head_nod - poly(crank)| {math.degrees(viol_p):.4f} deg, vs brentq table {math.degrees(viol_t):.4f} deg, "
    f"max |ear - poly(crank)| {math.degrees(viol_e):.4f} deg -> {'PASS' if okL else 'FAIL'} (limit 0.5 deg)")

# ------------------------------------------------------------------ D: stand still 3 s
log("=== Gate D: free model, zero-pose position targets, 3 s")
zero()
d.qpos[2] += 0.0005; mujoco.mj_forward(m, d)
torso = bid("torso")
nu = m.nu; fmax = np.zeros(nu); viol = 0.0
z0 = d.subtree_com[torso][2]; tilt_max = 0.0; comz_min = z0
hn, nc = qadr("head_nod"), qadr("nod_crank")
for k in range(int(3.0 / m.opt.timestep)):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)
    if k * m.opt.timestep > 0.05:
        fmax = np.maximum(fmax, np.abs(d.actuator_force))
    R3 = d.xmat[torso].reshape(3, 3)
    tilt_max = max(tilt_max, math.degrees(math.acos(np.clip(R3[2, 2], -1, 1))))
    comz_min = min(comz_min, d.subtree_com[torso][2])
    viol = max(viol, abs(d.qpos[hn] - PV(d.qpos[nc], cH)))
com = d.subtree_com[torso] * 1e3
tilt_end = math.degrees(math.acos(np.clip(d.xmat[torso].reshape(3, 3)[2, 2], -1, 1)))
fallen = tilt_max > 10.0 or (z0 - comz_min) * 1e3 > 20.0
# support polygon: convex hull of the 8 pad footprints (circles r 6 mm) at the end
ring = []
for i in pads:
    c = d.geom_xpos[i] * 1e3
    for t in np.linspace(0, 2 * math.pi, 32, endpoint=False):
        ring.append((c[0] + 6 * math.cos(t), c[1] + 6 * math.sin(t)))
ring = np.array(ring); H = ConvexHull(ring)
margin = min(-(eq[:2] @ com[:2] + eq[2]) for eq in H.equations)
log(f"  fallen: {'YES' if fallen else 'no'} (max torso tilt {tilt_max:.3f} deg, com height drop {(z0 - comz_min) * 1e3:.3f} mm;"
    f" fall = tilt > 10 deg or drop > 20 mm; tilt at 3 s {tilt_end:.3f} deg)")
log(f"  final com ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) mm; support polygon (hull of 8 pad circles) "
    f"x {ring[:, 0].min():.1f}..{ring[:, 0].max():.1f}, y {ring[:, 1].min():.1f}..{ring[:, 1].max():.1f} mm")
log(f"  com margin inside the support polygon: {margin:.1f} mm (nearest edge)")
log(f"  loop violation during standing: {math.degrees(viol):.4f} deg")
log(f"  contacts at the end: {d.ncon}")
log("  max |actuator force| over 0.05-3 s (N*m) and / stall torque:")
for i in range(nu):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i); st = m.actuator_forcerange[i][1]
    log(f"    {nm:14s} {fmax[i]:.4f}  / {st:.2f} = {fmax[i] / st:.3f}")
log("  at 3 s: joint angle error (deg) vs actuator_force / kp (deg) — servo compliance check:")
for i in range(nu):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i); ja_ = m.jnt_qposadr[m.actuator_trnid[i][0]]
    err = math.degrees(d.ctrl[i] - d.qpos[ja_]); kp = m.actuator_gainprm[i][0]
    if abs(err) > 0.01:
        log(f"    {nm:14s} error {err:+.3f}  force/kp {math.degrees(d.actuator_force[i] / kp):+.3f}")
okD = not fallen and margin > 0
log(f"  -> {'PASS' if okD else 'FAIL'}")

log("=== Summary: A %s, B %s, C %s, loop %s, D %s" % tuple("PASS" if x else "FAIL" for x in (okA, okB, okC, okL, okD)))
open(os.path.join(GD, "gate_log.txt"), "w").write("\n".join(LOG) + "\n")
