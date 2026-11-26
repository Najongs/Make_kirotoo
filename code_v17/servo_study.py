"""Joint torque / speed study for servo assignment.

Rigid-body model: one body per link (mass, CoM, inertia from mass_props). Legs: hip yaw (z), hip pitch (y),
knee (y), ankle pitch (y), ankle roll (x). No hip roll, so the robot walks by rocking: in single support the
whole body rolls about the stance ankle until the CoM is over the stance foot; the swing foot lifts by the roll.

Per time step: pelvis roll psi(t), pelvis x relative to the stance ankle, leg heights -> leg IK -> body poses.
Velocities/accelerations by finite differences. Joint torque = axis . sum over the bodies on the far side of the
joint (stance leg: everything above; swing leg / arms: everything distal) of the moment of their inertial +
gravity wrench. ZMP from the total wrench.

usage: python3 servo_study.py mass_build14_light.json
"""
import sys, json, itertools, numpy as np

G = np.array([0, 0, -9.81e3])      # mm/s2
# 11.1 V battery: stall torque (N m), no-load speed (rad/s)
SPEC = {"XM430-W350": (3.8, 43 * 2 * np.pi / 60), "XC430-W240": (1.75, 65 * 2 * np.pi / 60),
        "XL430-W250": (1.4, 52 * 2 * np.pi / 60)}
SMASS = {"XM430-W350": 90.0, "XC430-W240": 73.0, "XL430-W250": 65.2}
LEGJ = ["hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
LEGL = ["hip_yaw_link", "thigh", "shank", "ankle_link", "foot"]
SERVO_LINK = {"hip_yaw": "torso", "hip_pitch": "hip_yaw_link", "knee": "thigh", "ankle_pitch": "shank",
              "ankle_roll": "ankle_link"}      # link that carries the servo body of that joint
AX = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
JAX = {"hip_yaw": "z", "hip_pitch": "y", "knee": "y", "ankle_pitch": "y", "ankle_roll": "x"}
SOLE_Z = 0.0
FOOT_X = (-74.0, 60.0)
FOOT_HW = 35.0


def rot(axis, q):
    c, s = np.cos(q), np.sin(q)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Model:
    def __init__(self, mass, assign=None, arm_servo="XC430-W240", nod_servo="XC430-W240"):
        L = {k: dict(v) for k, v in mass["links"].items()}
        # fold the tiny mechanism links into their parents
        for small, parent in (("nod_crank", "head"), ("nod_rod", "head"), ("arm_crank", "torso"),
                              ("R_arm_rod", "torso"), ("L_arm_rod", "torso"), ("ears", "head"), ("ear_rod", "head")):
            if small in L:
                self._merge(L, parent, small)
        self.L = L
        if assign:
            self.reassign(assign, arm_servo, nod_servo)

    @staticmethod
    def _merge(L, parent, child):
        a, b = L[parent], L.pop(child)
        m = a["mass_g"] + b["mass_g"]
        ca, cb = np.array(a["com_world"]), np.array(b["com_world"])
        c = (a["mass_g"] * ca + b["mass_g"] * cb) / m
        I = np.zeros((3, 3))
        for mm, cc, Ii in ((a["mass_g"], ca, a["inertia_gmm2"]), (b["mass_g"], cb, b["inertia_gmm2"])):
            d = cc - c
            I += np.array(Ii) + mm * (d @ d * np.eye(3) - np.outer(d, d))
        a.update(mass_g=m, com_world=list(c), inertia_gmm2=I.tolist())

    def reassign(self, assign, arm_servo, nod_servo):
        """assign: {joint: servo type} for the leg joints. Model built with XM at knee/ankles, XC at hips."""
        base = {"hip_yaw": "XC430-W240", "hip_pitch": "XC430-W240", "knee": "XM430-W350",
                "ankle_pitch": "XM430-W350", "ankle_roll": "XM430-W350"}
        for s in "RL":
            for j, t in assign.items():
                d = SMASS[t] - SMASS[base[j]]
                ln = SERVO_LINK[j] if SERVO_LINK[j] == "torso" else f"{s}_{SERVO_LINK[j]}"
                self.L[ln]["mass_g"] += d     # CoM shift neglected (same housing)
        self.L["torso"]["mass_g"] += SMASS[arm_servo] - 73.0
        self.L["head"]["mass_g"] += SMASS[nod_servo] - 73.0

    def total(self):
        return sum(v["mass_g"] for v in self.L.values())


def leg_ik(dx, dz, l1=75.0, l2=75.0):
    """hip-pitch -> ankle vector (dx, dz) in the leg plane -> (q_hip, q_knee), knee forward."""
    d = np.hypot(dx, dz)
    d = min(d, l1 + l2 - 1e-6)
    c = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    q2 = np.arccos(np.clip(c, -1, 1))
    alpha = np.arctan2(-dx, -dz)
    q1 = alpha - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return q1, q2


def poses(M, ys_st, q_st, q_sw, P, R, arm):
    """World pose (R, CoM) of every body. q_* = (yaw, hp, kn, ap, ar)."""
    out = {}
    o_t = np.zeros(3)
    def place(name, Rw, pw, origin):
        v = M.L[name]
        c = np.array(v["com_world"]) - origin
        out[name] = (Rw, pw + Rw @ c)
    place("torso", R, P, o_t)
    place("head", R, P, o_t)
    for s, ang in (("R", arm[0]), ("L", arm[1])):
        o = np.array(M.L[f"{s}_arm"]["origin"])
        place(f"{s}_arm", R @ rot("y", ang), P + R @ o, o)
    joints = {}
    for s, ys, q in (("R" if ys_st < 0 else "L", ys_st, q_st), ("L" if ys_st < 0 else "R", -ys_st, q_sw)):
        Rw, pw = R, P.copy()
        prev = np.zeros(3)
        for j, ln, qi in zip(LEGJ, LEGL, q):
            o = np.array(M.L[f"{s}_{ln}"]["origin"])
            pw = pw + Rw @ (o - prev)
            prev = o
            Rw = Rw @ rot(JAX[j], qi)
            joints[f"{s}_{j}"] = (pw.copy(), (Rw @ AX[JAX[j]]))
            place(f"{s}_{ln}", Rw, pw, o)
    return out, joints


def gait(M, step, T, crouch, lift, psi_amp, n=200, arm_amp=np.radians(14), arms=True):
    """One step on the right leg (stance R, swing L), periodic. Returns time series."""
    ys = -40.0
    dt = T / n
    ts = np.arange(-2, n + 3) * dt
    H = 185.0 - 35.0 - crouch            # hip pitch above ankle, in the leg plane (vertical in pelvis frame)
    rec = []
    for t in ts:
        u = t / T
        psi = psi_amp * np.sin(np.pi * u)                # rolls onto the right foot (+x roll moves pelvis to -y)
        xr = -step / 2 + step * u                        # pelvis x relative to stance ankle
        q1, q2 = leg_ik(-xr, -H)
        q_st = (0.0, q1, q2, -(q1 + q2), -psi)
        # swing ankle in pelvis frame: from -step/2 to +step/2 (cycloid), extra lift by knee
        xs = -step / 2 + step * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
        zl = lift * (1 - np.cos(2 * np.pi * u)) / 2
        h1, h2 = leg_ik(xs, -(H - zl))
        q_sw = (0.0, h1, h2, -(h1 + h2), -psi)
        R = rot("x", psi)
        A = np.array([0.0, ys, 35.0])                    # stance ankle (world, fixed)
        a_p = np.array([-xr, ys, 35.0 + 0.0]) - np.array([0, 0, 0])
        # ankle in pelvis frame: hip pitch at z 185, ankle at 185 - H
        a_pel = np.array([-xr, ys, 185.0 - H])
        P = A - R @ a_pel
        ang = arm_amp * np.sin(np.pi * u - np.pi / 2) if arms else 0.0   # right arm forward when left leg forward
        bod, jts = poses(M, ys, q_st, q_sw, P, R, (-ang, ang))
        rec.append((t, bod, jts, dict(st=q_st, sw=q_sw), psi))
    return rec, dt


def dynamics(M, rec, dt, stance="R"):
    """Joint torques (N m), joint speeds (rad/s), ZMP (mm) at the inner samples."""
    names = list(rec[0][1].keys())
    pos = {b: np.array([r[1][b][1] for r in rec]) for b in names}
    Rm = {b: np.array([r[1][b][0] for r in rec]) for b in names}
    acc = {b: (pos[b][2:] - 2 * pos[b][1:-1] + pos[b][:-2]) / dt ** 2 for b in names}
    # angular velocity from R_{k+1} R_k^T
    def omega(Rs):
        w = []
        for k in range(len(Rs) - 1):
            D = Rs[k + 1] @ Rs[k].T
            w.append(np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2 * dt))
        return np.array(w)
    om = {b: omega(Rm[b]) for b in names}
    wmid = {b: (om[b][1:] + om[b][:-1]) / 2 for b in names}
    alp = {b: (om[b][1:] - om[b][:-1]) / dt for b in names}
    sw = "L" if stance == "R" else "R"
    leg_order = [f"{stance}_{l}" for l in LEGL]
    out = {}
    K = len(rec) - 2
    wrench = {}
    for b in names:
        m = M.L[b]["mass_g"] / 1000.0                    # kg; mm/s2 -> N*mm  (kg*mm/s2 = 1e-3 N) handled below
        I = np.array(M.L[b]["inertia_gmm2"]) / 1000.0     # kg mm2
        F = m * (acc[b] - G) * 1e-3                       # N
        tau = np.array([Rm[b][k + 1] @ I @ Rm[b][k + 1].T @ alp[b][k]
                        + np.cross(wmid[b][k], Rm[b][k + 1] @ I @ Rm[b][k + 1].T @ wmid[b][k])
                        for k in range(K)]) * 1e-6        # kg mm2/s2 = 1e-6 N m
        wrench[b] = (F, tau)
    def moment(bodies, k, p):
        s = np.zeros(3)
        for b in bodies:
            F, tau = wrench[b]
            r = (pos[b][k + 1] - p) * 1e-3
            s += np.cross(r, F[k]) + tau[k]
        return s
    tq = {}
    for i, j in enumerate(LEGJ):
        # stance: bodies above joint i = all bodies minus stance-leg links at index >= i
        above = [b for b in names if b not in leg_order[i:]]
        swing_distal = [f"{sw}_{l}" for l in LEGL[i:]]
        ts_, tw_ = [], []
        for k in range(K):
            p, a = rec[k + 1][2][f"{stance}_{j}"]
            ts_.append(a @ moment(above, k, p))
            p, a = rec[k + 1][2][f"{sw}_{j}"]
            tw_.append(a @ moment(swing_distal, k, p))
        tq[f"stance_{j}"] = np.array(ts_)
        tq[f"swing_{j}"] = np.array(tw_)
    for s in ("R", "L"):
        ts_ = []
        for k in range(K):
            p, a = rec[k + 1][1][f"{s}_arm"][1], None
        # arm swing torque about shoulder y (from the crank)
    # joint speeds
    qs = {}
    for key, idx in (("stance", "st"), ("swing", "sw")):
        Q = np.array([r[3][idx] for r in rec])
        Qd = (Q[2:] - Q[:-2]) / (2 * dt)
        for i, j in enumerate(LEGJ):
            qs[f"{key}_{j}"] = Qd[:, i]
    # ZMP
    zmp = []
    for k in range(K):
        Ft = sum(wrench[b][0][k] for b in names)
        Mo = moment(names, k, np.zeros(3))                 # about world origin (N m)
        # ground point p=(x,y,0): Mo - p x Ft has zero horizontal comp.
        x = (-Mo[1]) / Ft[2] * 1e3
        y = (Mo[0]) / Ft[2] * 1e3
        zmp.append((x, y))
    return tq, qs, np.array(zmp)


def run_case(M, case):
    rec, dt = gait(M, **case)
    tq, qs, zmp = dynamics(M, rec, dt)
    return tq, qs, zmp, rec


def static_cases(M):
    """Single support, quasi static: deep crouch and CoM over the toe / heel / side of the foot."""
    res = {}
    for nm, crouch, xr, psi in (("crouch", 40.0, 0.0, None), ("toe", 15.0, 55.0, None), ("heel", 15.0, -60.0, None)):
        ys = -40.0
        H = 150.0 - crouch
        q1, q2 = leg_ik(-xr, -H)
        # roll so that the CoM is laterally over the stance ankle (solve by iteration)
        psi_ = 0.0
        for _ in range(30):
            R = rot("x", psi_)
            a_pel = np.array([-xr, ys, 185.0 - H])
            P = np.array([0.0, ys, 35.0]) - R @ a_pel
            q_st = (0.0, q1, q2, -(q1 + q2), -psi_)
            bod, jts = poses(M, ys, q_st, q_st, P, R, (0, 0))
            m = sum(M.L[b]["mass_g"] for b in bod)
            cy = sum(M.L[b]["mass_g"] * bod[b][1][1] for b in bod) / m
            psi_ += (cy - ys) / 190.0
        rec = [(i, bod, jts, dict(st=q_st, sw=q_st), psi_) for i in range(5)]
        tq, qs, zmp = dynamics(M, rec, 0.01)
        res[nm] = ({k: abs(v[1]) for k, v in tq.items() if k.startswith("stance")}, zmp[1], np.degrees(psi_))
    return res


CASES = {
    "walk": dict(step=30.0, T=0.45, crouch=12.0, lift=8.0, psi_amp=np.radians(11)),
    "fast": dict(step=45.0, T=0.32, crouch=15.0, lift=10.0, psi_amp=np.radians(10)),
    "jog":  dict(step=55.0, T=0.25, crouch=18.0, lift=12.0, psi_amp=np.radians(8)),
}


def util(tau, w, servo):
    ts, w0 = SPEC[servo]
    return np.max(np.abs(tau) / ts + np.abs(w) / w0), np.max(np.abs(tau)), np.sqrt(np.mean(tau ** 2))


def tune_psi(M, case):
    """smallest roll amplitude that keeps the ZMP inside the stance foot (y) in mid single support"""
    best = None
    for deg in np.arange(3, 18, 0.5):
        c = dict(case, psi_amp=np.radians(deg))
        rec, dt = gait(M, n=80, **c)
        _, _, zmp = dynamics(M, rec, dt)
        k = len(zmp)
        mid = zmp[k // 4: 3 * k // 4]
        err = np.max(np.abs(mid[:, 1] + 40.0))
        if best is None or err < best[1]:
            best = (deg, err)
    return best


if __name__ == "__main__":
    mass = json.load(open(sys.argv[1]))
    M = Model(mass)
    print("mass", round(M.total()))
    for nm, c in CASES.items():
        deg, err = tune_psi(M, c)
        CASES[nm]["psi_amp"] = np.radians(deg)
        print(nm, "psi", deg, "zmp err", round(err, 1))
    for nm, c in CASES.items():
        tq, qs, zmp, rec = run_case(M, c)
        print(f"--- {nm}: zmp x {zmp[:,0].min():.0f}..{zmp[:,0].max():.0f}  y {zmp[:,1].min():.0f}..{zmp[:,1].max():.0f}")
        for k in tq:
            print(f"   {k:20s} peak {np.max(np.abs(tq[k])):5.2f} Nm  rms {np.sqrt(np.mean(tq[k]**2)):5.2f}  "
                  f"wmax {np.max(np.abs(qs[k])):5.2f} rad/s")
    for k, v in static_cases(M).items():
        print(k, {a: round(b, 2) for a, b in v[0].items()}, "zmp", np.round(v[1], 0), "psi", round(v[2], 1))
