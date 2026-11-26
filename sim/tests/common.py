"""Shared helpers for sim/tests: model loading, servo tables, stance-foot kinematics, foot-rooted inverse dynamics,
servo_study trajectories (code_v17, imported read-only)."""
import json, math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")
import numpy as np
import mujoco

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(SIM, "..", "code_v17")
sys.path.insert(0, CODE)
RES = os.path.join(SIM, os.environ.get("KITORU_RES", "results")); os.makedirs(RES, exist_ok=True)
# KITORU_MODEL = model stem in sim/model (default kitoru_v17), KITORU_MASSF = mass json for servo_study,
# KITORU_RES = results subfolder (default results/)
_STEM = os.environ.get("KITORU_MODEL", "kitoru_v17")
XML = os.path.join(SIM, "model", f"{_STEM}.xml")
XMLF = os.path.join(SIM, "model", f"{_STEM}_fixed.xml")
MASSF = os.environ.get("KITORU_MASSF", os.path.join(CODE, "mass_build17_light.json"))
LOOP = json.load(open(os.path.join(SIM, "model", "closed_loop_fit.json")))
PV = np.polynomial.polynomial.polyval

LEGJ = ["hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
SERVO_OF = {"hip_yaw": "XC430-W240", "hip_pitch": "XM430-W350", "knee": "XM430-W350", "ankle_pitch": "XC430-W240",
            "ankle_roll": "XM430-W350", "arm_swing": "XC430-W240", "nod_crank": "XL430-W250"}
# (stall N*m, no-load rad/s): the model / handover table, and the ROBOTIS e-manual (11.1 V)
SPEC_MODEL = {"XM430-W350": (3.8, 4.5), "XC430-W240": (1.75, 6.81), "XL430-W250": (1.4, 5.45)}
SPEC_EMAN = {"XM430-W350": (3.8, 43 * 2 * math.pi / 60), "XC430-W240": (1.7, 65 * 2 * math.pi / 60),
             "XL430-W250": (1.4, 57 * 2 * math.pi / 60)}


def load(fixed=False):
    m = mujoco.MjModel.from_xml_path(XMLF if fixed else XML)
    return m, mujoco.MjData(m)


def rigid_copy(m, keep_armature=False):
    """same model with all constraints (contacts, loop equalities, limits) off and without damping / friction loss
    (damping = back-EMF, already in the |w|/w0 term of u); armature removed too unless keep_armature: pure
    rigid-body inverse dynamics, the assumption of servo_study.py."""
    r = mujoco.MjModel.from_xml_path(XMLF if m.nq == 15 else XML)
    r.dof_damping[:] = 0; r.dof_frictionloss[:] = 0
    if not keep_armature:
        r.dof_armature[:] = 0
    r.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return r


def jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def qa(m, n):
    return m.jnt_qposadr[jid(m, n)]


def va(m, n):
    return m.jnt_dofadr[jid(m, n)]


def bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def aid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


def gname(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"


def body_of_geom(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])


def loop_qpos(alpha):
    """head_nod, ear_axle (rad) for a crank angle alpha (rad), via the fitted polynomials"""
    return PV(alpha, LOOP["polycoef_head"]), PV(alpha, LOOP["polycoef_ear"])


def loop_slopes(alpha):
    dH = np.polynomial.polynomial.polyder(LOOP["polycoef_head"]); dE = np.polynomial.polynomial.polyder(LOOP["polycoef_ear"])
    return PV(alpha, dH), PV(alpha, dE)


def set_joints(m, q, ang):
    """ang: {joint name: rad}; head_nod/ear_axle follow nod_crank if not given"""
    for k, v in ang.items():
        q[qa(m, k)] = v
    if "nod_crank" in ang and "head_nod" not in ang:
        h, e = loop_qpos(ang["nod_crank"]); q[qa(m, "head_nod")] = h; q[qa(m, "ear_axle")] = e


def quat_of(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


def torso_for_stance(m, d, ang, stance, foot_pos=None, foot_R=np.eye(3)):
    """qpos (free model) with joint angles `ang` and the torso placed so that the stance foot body sits at
    foot_pos / foot_R (default: its zero-pose place, flat on the floor)."""
    fb = bid(m, f"{stance}_foot")
    q = m.qpos0.copy(); q[3:7] = [1, 0, 0, 0]; set_joints(m, q, ang)
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    pr, Rr = d.xpos[fb].copy(), d.xmat[fb].reshape(3, 3).copy()        # foot pose with the torso at identity
    if foot_pos is None:
        d2 = mujoco.MjData(m); d2.qpos[:] = m.qpos0; d2.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d2); foot_pos = d2.xpos[fb].copy()
    Rt = foot_R @ Rr.T
    q[0:3] = foot_pos - Rt @ pr; q[3:7] = quat_of(Rt)
    return q


def fd(m, Q, dt):
    """central finite differences on qpos rows -> qvel, qacc for rows 1..n-2"""
    n = len(Q); nv = m.nv
    V = np.zeros((n - 2, nv)); A = np.zeros((n - 2, nv))
    for k in range(1, n - 1):
        vp = np.zeros(nv); vm = np.zeros(nv); vc = np.zeros(nv)
        mujoco.mj_differentiatePos(m, vp, dt, Q[k], Q[k + 1])
        mujoco.mj_differentiatePos(m, vm, dt, Q[k - 1], Q[k])
        V[k - 1] = 0.5 * (vp + vm); A[k - 1] = (vp - vm) / dt
    return V, A


def inverse_foot_root(m, d, q, v, a, stance):
    """mj_inverse with the stance foot as the root: the 6 free-joint components of qfrc_inverse are supplied by a
    ground wrench on the stance foot (applied at the foot body origin). Returns joint torques (nv,), the wrench
    (force N, moment N*m about the foot origin) and the centre of pressure on z=0 (m)."""
    d.qpos[:] = q; d.qvel[:] = v; d.qacc[:] = a
    mujoco.mj_inverse(m, d)
    fb = bid(m, f"{stance}_foot")
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jp, jr, fb)
    J = np.vstack([jp, jr])
    w = np.linalg.solve(J[:, :6].T, d.qfrc_inverse[:6])
    tau = d.qfrc_inverse - J.T @ w
    f, n0 = w[:3], w[3:]; p0 = d.xpos[fb]
    h = p0[2]
    rx = (n0[1] + h * f[0]) / f[2]; ry = (h * f[1] - n0[0]) / f[2]
    cop = np.array([p0[0] - rx, p0[1] - ry, 0.0])
    return tau, w, cop


# ---------------------------------------------------------------- servo_study trajectories
import servo_study as S


def ss_model(assigned=False):
    mass = json.load(open(MASSF))
    if not assigned:
        return S.Model(mass)
    assign = {j: SERVO_OF[j] for j in LEGJ}
    return S.Model(mass, assign, arm_servo="XL430-W250", nod_servo="XL430-W250")   # as servo_opt.py builds it


_PSI = {}
def tuned_cases():
    """servo_study CASES with psi_amp from tune_psi (mass_build17_light.json, unassigned model as servo_opt does)"""
    if not _PSI:
        M0 = ss_model(False)
        for nm, c in S.CASES.items():
            deg, err = S.tune_psi(M0, c)
            _PSI[nm] = (deg, err)
    out = {}
    for nm, c in S.CASES.items():
        out[nm] = dict(c, psi_amp=np.radians(_PSI[nm][0]))
    return out, dict(_PSI)


def gait_angles(case, u, scale=1.0, stance="R"):
    """servo_study.gait() joint angles at phase u in [0,1] of one step, amplitude-scaled; stance 'R' as in
    servo_study, 'L' = mirror (roll/yaw negated). Returns ({joint: rad}, torso roll rad)."""
    step = case["step"] * scale; psi_amp = case["psi_amp"] * scale; lift = case["lift"] * scale
    arm_amp = np.radians(14) * scale
    H = 185.0 - 35.0 - case["crouch"]
    psi = psi_amp * np.sin(np.pi * u)
    xr = -step / 2 + step * u
    q1, q2 = S.leg_ik(-xr, -H)
    xs = -step / 2 + step * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    zl = lift * (1 - np.cos(2 * np.pi * u)) / 2
    h1, h2 = S.leg_ik(xs, -(H - zl))
    ang = arm_amp * np.sin(np.pi * u - np.pi / 2)
    sw = "L" if stance == "R" else "R"; sg = 1.0 if stance == "R" else -1.0
    out = {}
    for side, (a, b) in ((stance, (q1, q2)), (sw, (h1, h2))):
        out.update({f"{side}_hip_yaw": 0.0, f"{side}_hip_pitch": a, f"{side}_knee": b,
                    f"{side}_ankle_pitch": -(a + b), f"{side}_ankle_roll": -sg * psi})
    out["R_arm_swing"] = -sg * ang; out["L_arm_swing"] = sg * ang
    return out, sg * psi


def crouch_angles(crouch):
    H = 150.0 - crouch
    q1, q2 = S.leg_ik(0.0, -H)
    out = {}
    for s in "RL":
        out.update({f"{s}_hip_yaw": 0.0, f"{s}_hip_pitch": q1, f"{s}_knee": q2, f"{s}_ankle_pitch": -(q1 + q2),
                    f"{s}_ankle_roll": 0.0})
    return out


def jdump(obj, name):
    def conv(x):
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, np.ndarray): return conv(x.tolist())
        return x
    json.dump(conv(obj), open(os.path.join(RES, name), "w"), indent=1, ensure_ascii=False)


def korean_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = os.path.expanduser("~/.local/share/fonts/NotoSansKR.ttf")
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def foot_support(m, side="R"):
    """zero-pose support outlines of one foot (world mm, 2D): convex hull of its pad circles and of the sole
    bottom (collision hull vertices of <side>_foot_sole below z 8.5 mm)"""
    from scipy.spatial import ConvexHull
    d = mujoco.MjData(m); d.qpos[:] = m.qpos0
    if m.nq == 22:
        d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    pads = []
    for i in range(m.ngeom):
        if gname(m, i).startswith(f"{side}_foot_pad"):
            c = d.geom_xpos[i] * 1e3; r = m.geom_size[i][0] * 1e3
            pads += [(c[0] + r * math.cos(t), c[1] + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"col_{side}_foot_sole")
    mid = m.geom_dataid[i]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = (m.mesh_vert[a:a + n] @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]) * 1e3
    sole = V[V[:, 2] < 8.5][:, :2]
    P, S_ = np.array(pads), np.array(sole)
    return P[ConvexHull(P).vertices], S_[ConvexHull(S_).vertices]


def hull_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def hull_margin(poly, p):
    """signed distance of p to the boundary of convex polygon poly (positive inside)"""
    from scipy.spatial import ConvexHull
    H = ConvexHull(poly)
    return min(-(eq[:2] @ np.asarray(p)[:2] + eq[2]) for eq in H.equations)
