"""Kitoru v17 design (code_v17/build17.py) -> MuJoCo MJCF.

Run:  cd sim && uv run python export_mjcf.py
      v21 (elbow arm, head nod -20 .. +28, all 17 servos XL430-W250-T):
      cd sim && KITORU_VER=21 KITORU_MASS=../out21_arm/mass_build21_arm_light.json .venv/bin/python export_mjcf.py
      v22 (v21 kit after the durability passes, every link re-voxelised, joint ranges from the shape limits):
      cd sim && KITORU_VER=22 KITORU_MASS=model/mass_kitoru_v22_light.json .venv/bin/python export_mjcf.py
Out:  model/kitoru_v17.xml (free-floating torso), model/kitoru_v17_fixed.xml (torso welded, for later),
      model/meshes/*.obj, model/closed_loop_fit.json, model/export_log.txt

Rules (see gates/ for the checks):
- units m / kg / rad, every body frame is world-aligned (identity), body pos = child origin - parent origin,
  joint pos = joint point - body origin, meshes are world mm minus link origin with scale 0.001.
- mass / com / inertia only from code_v17/mass_build17_light.json (inertia: g*mm^2 about the com, world axes).
- nod_rod -> torso, ear_rod -> head, R/L_hand_flap -> R/L_arm are merged (parallel-axis) and welded
  (v21: no hand flaps; R/L_forearm hang on the new R/L_elbow joints).
- head_nod and ear_axle are passive hinges tied to nod_crank by <equality joint> polynomials fitted to the
  brentq closed-loop solution of sweep17.py / render17.py (same constants).
"""
import json, math, os, sys
import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code_v17")
sys.path.insert(0, CODE)
# version / variant selection (defaults = v17):
#   KITORU_VER=18 -> render18 / build18 (KITORU_THETA picks the foot cut), mass file and output names follow
#   KITORU_MASS  = mass json path (default code_v17/mass_build<ver>_light.json)
#   KITORU_TAG   = output name stem (default kitoru_v<ver>), meshes go to model/meshes (v17) or model/meshes_<tag>
#   KITORU_VER=21 -> render21 / build21_arm, all 17 actuators XL430-W250-T (ALLXL below)
#   KITORU_VER=22 -> render22 (= render21 on the current build21_arm), mass json already has every servo at XL430 65.2 g
#                    (no swap), leg pitch ranges from the shape limits (RANGE_V22 below)
import importlib
VER = os.environ.get("KITORU_VER", "17")
R17 = importlib.import_module(f"render{VER}")   # sets BUILD=build<ver>, imports render6 (GEOM) and fixes the chains
r = R17.r
TAG = os.environ.get("KITORU_TAG", f"kitoru_v{VER}")
import shapes, geom4
# layout A hip roll (build20): the legs reach below z 0 by LEG_EXTRA_MM, so the free model starts lifted by that much
# (v21: there is no build21 module, the build is the one render<ver> put in BUILD -> r.build5)
LIFT = float(getattr(sys.modules.get(f"build{VER}"), "LIFT_MM", getattr(r.build5, "LIFT_MM", 0.0)))
ALLXL = int(VER) >= 21       # owner 2026-11-27: every servo is an XL430-W250-T (17: legs 12, shoulders 2, elbows 2, nod 1)
from build import IDX, UVI

OUT = os.path.join(HERE, "model")
MESHSUB = "meshes" if TAG == "kitoru_v17" else f"meshes_{TAG}"
MESHDIR = os.path.join(OUT, MESHSUB)
os.makedirs(MESHDIR, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

links, joints = r.links, r.joints
J = {j["name"]: j for j in joints}
MASSF = os.environ.get("KITORU_MASS", os.path.join(CODE, f"mass_build{VER}_light.json"))
MASS = json.load(open(MASSF))["links"]

# ------------------------------------------------------------------ tree
MERGE = {"nod_rod": "torso", "ear_rod": "head", "R_hand_flap": "R_arm", "L_hand_flap": "L_arm"}
MERGE = {k: v for k, v in MERGE.items() if k in links}      # v21 elbow arm: no hand flaps
BODIES = ["torso",
          "R_hip_yaw_link", "R_thigh", "R_shank", "R_ankle_link", "R_foot",
          "L_hip_yaw_link", "L_thigh", "L_shank", "L_ankle_link", "L_foot",
          "R_arm", "L_arm", "head", "nod_crank", "ears"]
PARENT = {j["child"]: j["parent"] for j in joints}
JOINT_OF = {j["child"]: j["name"] for j in joints}
LEG_BITS = {f"{S}_{n}" for S in "RL" for n in ("shank", "ankle_link", "foot")}
for S_ in "RL":                 # v20: hip roll link between the hip yaw link and the thigh
    if f"{S_}_hip_roll_link" in links and f"{S_}_hip_roll_link" not in BODIES:
        BODIES.insert(BODIES.index(f"{S_}_hip_yaw_link") + 1, f"{S_}_hip_roll_link")
    if f"{S_}_forearm" in links and f"{S_}_forearm" not in BODIES:   # v21: elbow -> forearm under the upper arm
        BODIES.insert(BODIES.index(f"{S_}_arm") + 1, f"{S_}_forearm")
assert set(BODIES) | set(MERGE) == set(links) == set(MASS)


def origin(ln):
    return np.array(links[ln]["origin"], float)


# ------------------------------------------------------------------ servos
# 11.1 V values given by the owner (handover doc 3.1). XC430 stall: e-manual lists 1.7 N*m @11.1 V, the design
# uses 1.75 -> kept 1.75 (difference reported). Gear ratios from the ROBOTIS e-manual spec tables:
# XM430-W350 353.5:1 (coreless), XC430-W240 245.22:1 (coreless), XL430-W250 258.5:1 (cored).
SERVO = {
    "XM430-W350": dict(stall=3.8, speed=4.5, kp=17.5, arm=0.045),
    "XC430-W240": dict(stall=1.75, speed=6.81, kp=7.0, arm=round(0.045 * (245.22 / 353.5) ** 2, 4)),
    "XL430-W250": dict(stall=1.4, speed=5.45, kp=6.4, arm=round(0.045 * (258.5 / 353.5) ** 2, 4)),
}
SERVO_NOTE = """
  Servo model (position-control mode): native <position> actuator, forcerange = +-stall torque,
  joint damping = stall / no-load speed, armature = reflected rotor inertia.
  - stall / no-load speed @11.1 V: owner's table (handover 3.1). e-manual differs on two values, owner's kept:
    XC430-W240 stall 1.7 N*m (table 1.75); XL430-W250 no-load 57 rpm = 5.97 rad/s (table 5.45 -> damping 0.257, not 0.235).
  - kp [estimate]: Position P gain default XM 800 / XC 700 (XL assumed 800), KPP = TBL/128, PWM limit 885,
    651.9 pulse/rad -> XM 17.5, XC 7.0, XL 6.4 N*m/rad.
  - armature XM430: mujoco_menagerie robotis_op3/op3.xml (sha eea4c74) default joint armature 0.045
    (same file: damping 1.084, frictionloss 0.03, kp 21.1, forcerange +-5; frictionloss not used here).
  - armature XC/XL [estimate]: 0.045 * (gear / 353.5)^2 with e-manual gear ratios XC 245.22, XL 258.5.
    Assumes the same rotor inertia as XM (coreless). XL430 has a cored motor (heavier rotor) -> likely low.
"""
if ALLXL:
    # v21: XL430-W250-T no-load 57 rpm @11.1 V = 5.97 rad/s (e-manual, sim/walk/allxl XL430e) instead of the 5.45 above
    SERVO["XL430-W250"] = dict(SERVO["XL430-W250"], speed=round(57 * 2 * math.pi / 60, 2))
    SERVO_NOTE = """
  Servo model (position-control mode): native <position> actuator, forcerange = +-stall torque,
  joint damping = stall / no-load speed, armature = reflected rotor inertia.
  - all 17 servos XL430-W250-T (owner 2026-11-27; build20 joint table still names XM430 / XC430 on the legs).
  - stall 1.4 N*m, no-load 57 rpm = 5.97 rad/s @11.1 V (ROBOTIS e-manual; sim/walk/allxl XL430e) -> damping 0.2345.
  - kp 6.4 N*m/rad [estimate]: Position P gain 800 (assumed), KPP = TBL/128, PWM limit 885, 651.9 pulse/rad.
  - armature 0.0241 [estimate]: 0.045 (menagerie robotis_op3 XM430) * (258.5 / 353.5)^2. XL430 has a cored motor
    (heavier rotor) -> likely low; measure on the real servo.
  - mass: the mass json has the leg servos as XM430 90 g / XC430 73 g; each is swapped to XL430 65.2 g (horn +
    cables, sim/walk/kw_common SERVO_G) as a point mass at that servo's com (parallel axis; the servo's own inertia
    change is ignored).
"""
    if int(VER) >= 22:
        SERVO_NOTE = SERVO_NOTE.replace("""  - mass: the mass json has the leg servos as XM430 90 g / XC430 73 g; each is swapped to XL430 65.2 g (horn +
    cables, sim/walk/kw_common SERVO_G) as a point mass at that servo's com (parallel axis; the servo's own inertia
    change is ignored).""", """  - mass: sim/tools/mass_v22.py voxelises every link of the current geometry (profile light; foot soles and shoulder
    horn adapters 100 % infill as the print kit says) with every servo at XL430 65.2 g (57.2 + 8 horn / screws /
    cable) spread over its case envelope (inertia included); no swap here.
  - ranges: hip pitch -117 / +113, knee 0 / 117, ankle pitch -75 / +30 from the shape limits (RANGE_V22).""")
ACT = {}            # joint -> servo name
for j in joints:
    for s in SERVO:
        if j["actuator"].startswith(s):
            ACT[j["name"]] = "XL430-W250" if ALLXL else s
assert len(ACT) >= (17 if ALLXL else 13), ACT


# ------------------------------------------------------------------ mass
SERVO_SWAP = []     # v21: (link, part, delta g) of the XM / XC -> XL430 mass swap
if ALLXL:
    G_XL = 65.2
    for p in json.load(open(MASSF))["parts"]:
        if p["kind"] != "servo" or p["part"].endswith("XL430"):
            continue
        L_ = MASS[p["link"]]; dm = G_XL - p["mass"]; pc = np.array(p["com"], float)
        m0, c0, I0 = L_["mass_g"], np.array(L_["com_world"], float), np.array(L_["inertia_gmm2"], float)
        m1 = m0 + dm; c1 = (m0 * c0 + dm * pc) / m1
        d, e = c0 - c1, pc - c1
        I1 = I0 + m0 * (d @ d * np.eye(3) - np.outer(d, d)) + dm * (e @ e * np.eye(3) - np.outer(e, e))
        assert np.all(np.linalg.eigvalsh(I1) > 0), (p["part"], I1)
        L_.update(mass_g=m1, com_world=list(c1), inertia_gmm2=I1.tolist())
        SERVO_SWAP.append((p["link"], p["part"], round(dm, 1)))
    SERVO_SWAP = [x for x in SERVO_SWAP if abs(x[2]) > 1e-9]     # v22 mass json: servos are 65.2 g already
    assert len(SERVO_SWAP) == (0 if int(VER) >= 22 else 10), SERVO_SWAP


def inertial(ln):
    """mass (g), com (world mm), inertia (g mm^2 about com, world axes) of a body incl. merged links."""
    parts = [ln] + [k for k, v in MERGE.items() if v == ln]
    m = sum(MASS[p]["mass_g"] for p in parts)
    c = sum(MASS[p]["mass_g"] * np.array(MASS[p]["com_world"]) for p in parts) / m
    I = np.zeros((3, 3))
    for p in parts:
        mp = MASS[p]["mass_g"]; d = np.array(MASS[p]["com_world"]) - c
        I += np.array(MASS[p]["inertia_gmm2"]) + mp * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, I


# ------------------------------------------------------------------ closed loop (render17 = sweep17 math)
NOD = (float(J["head_nod"]["lower_deg"]), float(J["head_nod"]["upper_deg"])) if ALLXL else (-10.0, 20.0)
betas = np.round(np.arange(NOD[0], NOD[1] + 1e-9, 0.25), 4)
tab = [R17.nod_pose(float(b)) for b in betas]
al = np.array([t["nod_crank"] for t in tab]); ps = np.array([t["ear_axle"] for t in tab])
alr, br, psr = np.radians(al), np.radians(betas), np.radians(ps)
cf_head = np.polyfit(alr, br, 4)[::-1]          # head_nod = sum c_i * crank^i   (rad)
cf_ear = np.polyfit(alr, psr, 4)[::-1]
res_h = np.degrees(np.polynomial.polynomial.polyval(alr, cf_head) - br)
res_e = np.degrees(np.polynomial.polynomial.polyval(alr, cf_ear) - psr)
dB = np.gradient(betas, al); dP = np.gradient(ps, al)
LOOP = dict(beta_deg=[float(x) for x in betas], crank_deg=[float(x) for x in al], ear_deg=[float(x) for x in ps],
            polycoef_head=[float(x) for x in cf_head], polycoef_ear=[float(x) for x in cf_ear],
            max_resid_head_deg=float(np.abs(res_h).max()), max_resid_ear_deg=float(np.abs(res_e).max()),
            dbeta_dalpha=[float(dB.min()), float(dB.max())], dpsi_dalpha=[float(dP.min()), float(dP.max())])
json.dump(LOOP, open(os.path.join(OUT, "closed_loop_fit.json" if TAG == "kitoru_v17" else f"closed_loop_fit_{TAG}.json"), "w"), indent=1)
CRANK_RANGE = (float(al.min()), float(al.max())); EAR_RANGE = (float(ps.min()), float(ps.max()))
log(f"closed loop: crank {CRANK_RANGE[0]:.3f}..{CRANK_RANGE[1]:.3f} deg for head {NOD[0]:g}..{NOD[1]:g}, ear {EAR_RANGE[0]:.3f}..{EAR_RANGE[1]:.3f} deg")
log(f"  polyfit deg4 residual: head {LOOP['max_resid_head_deg']:.2e} deg, ear {LOOP['max_resid_ear_deg']:.2e} deg; "
    f"dbeta/dalpha {dB.min():.4f}..{dB.max():.4f}, dpsi/dalpha {dP.min():.4f}..{dP.max():.4f}")
log(f"  c0 head {cf_head[0]:.2e} rad, c0 ear {cf_ear[0]:.2e} rad (should be ~0: beta=0 <-> crank=0)")
RANGE = {n: (J[n]["lower_deg"], J[n]["upper_deg"]) for n in J}
RANGE["head_nod"] = NOD; RANGE["nod_crank"] = CRANK_RANGE; RANGE["ear_axle"] = EAR_RANGE
# v22 (owner 2026-11-28): leg pitch ranges from the shape limits (out20/LEG_PRINT_KIT_R.md; re-run on this build in
# sim/walk/ranges/results/v22 and sim/tests/shape_contact_v22.py: hip pitch -117.6 / +114.5 shorts <-> thigh, knee 117.6
# thigh frame <-> shank frame (105 .. 114 with the hip forward: hip roll link <-> ankle roll servo), ankle pitch -76.4 /
# +29.6 at roll 0) with about 1 deg margin; at ankle roll +-20 the foot reaches the shank frame at pitch -52 / +26 (the
# rl model collides there, see kitoru-rl assets/kitoru_v22/SOURCE.md). hip yaw / hip roll / ankle roll stay at the
# joints.csv values (shape: yaw -25.4 / +56, roll out -7.2 with the arm at 0 (forearm <-> thigh) / in 6 with the other leg
# at 0, ankle roll -31.6 / +26.9).
RANGE_V22 = {"hip_pitch": (-117.0, 113.0), "knee": (0.0, 117.0), "ankle_pitch": (-75.0, 30.0)}
if int(VER) >= 22:
    for S_ in "RL":
        for k, v in RANGE_V22.items():
            RANGE[f"{S_}_{k}"] = v
    log(f"v22 ranges: {RANGE_V22} (both legs)")


# ------------------------------------------------------------------ tessellation
def strip_tris(loops):
    """Exact cap triangulation for outline + holes (even-odd): cut the region into horizontal strips at every
    vertex height; inside a strip the crossing edges pair up into trapezoids. Used when ear clipping fails
    (render6.ear_clip stalls on the duplicated bridge vertices that bridge_holes inserts)."""
    E = []
    for lp in loops:
        n = len(lp)
        for i in range(n):
            p, q = lp[i], lp[(i + 1) % n]
            if abs(p[1] - q[1]) > 1e-12:
                E.append((p, q) if p[1] < q[1] else (q, p))
    vs = sorted(set(round(p[1], 9) for lp in loops for p in lp))
    T = []
    for v0, v1 in zip(vs[:-1], vs[1:]):
        vm = 0.5 * (v0 + v1)
        xs = []
        for p, q in E:
            if p[1] < vm < q[1]:
                t0 = (v0 - p[1]) / (q[1] - p[1]); t1 = (v1 - p[1]) / (q[1] - p[1])
                xs.append((p[0] + t0 * (q[0] - p[0]) + p[0] + t1 * (q[0] - p[0]),
                           p[0] + t0 * (q[0] - p[0]), p[0] + t1 * (q[0] - p[0])))
        xs.sort()
        for (_, a0, a1), (_, b0, b1) in zip(xs[0::2], xs[1::2]):
            T += [[(a0, v0), (b0, v0), (b1, v1)], [(a0, v0), (b1, v1), (a1, v1)]]
    return T


def tri_area(t):
    return 0.5 * abs((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]))


def cap_tris(outer, holes):
    """Triangles of one prism cap (render6.bridge_holes + ear_clip, strip fallback), plus area check."""
    outer = np.asarray(outer, float)
    k = int(np.argmin(np.ptp(outer, axis=0)))
    axis = "xyz"[k]; iu, iv = UVI[axis]; a = outer[0, k]
    o2 = [(p[iu], p[iv]) for p in outer]; h2 = [[(p[iu], p[iv]) for p in np.asarray(h, float)] for h in holes]
    want = abs(shapes.area(o2)) - sum(abs(shapes.area(h)) for h in h2)
    pts, tris = r.ear_clip(r.bridge_holes(o2, h2))
    T2 = [[pts[i], pts[j], pts[l]] for i, j, l in tris]
    got = sum(tri_area(t) for t in T2)
    used = "ear"
    if abs(want - got) > 1e-3 * max(want, 1.0):
        T2 = strip_tris([o2] + h2); used = "strip"
        got = sum(tri_area(t) for t in T2)
    def P(uv):
        q = [0.0, 0.0, 0.0]; q[iu], q[iv], q[k] = uv[0], uv[1], a; return q
    T = [[P(x) for x in t] for t in T2]
    return np.array(T, float).reshape(-1, 3, 3), want, got, used


ORIENT = {"kept": 0, "flipped": 0, "two_sided": 0}
def orient(s, T):
    """render6 triangles have mixed winding; MuJoCo culls back faces. Point each triangle outward by probing the
    solid (geom4.inside) just in front of / behind its centroid; where the probe is ambiguous keep both sides."""
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); nn = np.linalg.norm(N, axis=1)
    ok = nn > 1e-12; T, N = T[ok], N[ok] / nn[ok, None]
    C = T.mean(1)
    eps = 0.03 + {"cyl": lambda: 0.006 * s[1][2], "scap": lambda: 0.004 * s[1][2]}.get(s[0], lambda: 0.0)()
    pin, pout = geom4.inside(s, C + eps * N), geom4.inside(s, C - eps * N)
    flip = pin & ~pout; amb = pin == pout
    ORIENT["flipped"] += int(flip.sum()); ORIENT["two_sided"] += int(amb.sum()); ORIENT["kept"] += int((~flip & ~amb).sum())
    T = T.copy(); T[flip] = T[flip][:, ::-1]
    return np.concatenate([T, T[amb][:, ::-1]]) if amb.any() else T


def is_pad(ln, pn, s):
    return pn.endswith("_foot_sole") and s[0] == "cyl" and abs(s[1][2] - 6) < 1e-9 and abs(s[1][3] - 6) < 1e-9 \
        and tuple(s[1][1]) == (0, 0, 1) and abs(s[1][0][2]) < 1e-9


# per (body, rgb) visual triangles, per (body, part) hull points, pads
VIS, HULL, PADS = {}, {}, {}
cap_bad = []
cap_used = {}
for ln, L in links.items():
    body = MERGE.get(ln, ln)
    for pn, sols in L["parts"].items():
        if "joint_disc" in pn:
            continue
        for s in sols:
            if s[0] == "cyl" and tuple(s[2]) == tuple(r.DISC):
                continue
            if is_pad(ln, pn, s):
                PADS.setdefault(body, []).append(np.array(s[1][0], float))
                VIS.setdefault((body, tuple(s[2])), []).append(orient(s, r.tris_of(s)[0]))
                continue
            T, rgb, caps = r.tris_of(s)
            tl = [T] if len(T) else []
            for outer, holes in caps:
                Tc, want, got, used = cap_tris(outer, holes)
                cap_used[used] = cap_used.get(used, 0) + 1
                if abs(want - got) > 1e-3 * max(want, 1.0):
                    cap_bad.append((ln, pn, round(want, 2), round(got, 2)))
                tl.append(Tc)
            if not tl:
                continue
            T = np.concatenate(tl)
            HULL.setdefault((body, ln, pn), []).append(T.reshape(-1, 3))
            VIS.setdefault((body, tuple(rgb)), []).append(orient(s, T))
log(f"visual triangle winding: {ORIENT}")
if SERVO_SWAP:
    log(f"servo mass swap -> XL430 65.2 g: {SERVO_SWAP}; total {sum(L['mass_g'] for L in MASS.values()):.1f} g")
log(f"caps triangulated: {cap_used}; caps whose triangle area != outline-holes area: {len(cap_bad)}", cap_bad[:10])


def write_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write("v %.4f %.4f %.4f\n" % tuple(v))
        for t in F:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


def weld(T):
    P = T.reshape(-1, 3)
    key = np.round(P, 4)
    V, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V, F


def hull_mesh(P):
    P = np.unique(np.round(P, 4), axis=0)
    try:
        h = ConvexHull(P)
    except Exception:
        h = ConvexHull(P, qhull_options="QJ")
    V = P[h.vertices]; idx = {v: i for i, v in enumerate(h.vertices)}
    c = V.mean(0); F = []
    for simp, eq in zip(h.simplices, h.equations):
        t = [idx[x] for x in simp]
        n = np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])
        if n @ eq[:3] < 0:
            t = [t[0], t[2], t[1]]
        F.append(t)
    return V, np.array(F)


def rgba_name(rgb):
    return "c%02x%02x%02x" % tuple(int(round(255 * x)) for x in rgb)


MESH_ASSETS = []          # (name, file)
VIS_GEOMS = {b: [] for b in BODIES}
COL_GEOMS = {b: [] for b in BODIES}
MATS = {}
for (body, rgb), Ts in VIS.items():
    T = np.concatenate(Ts) - origin(body)
    V, F = weld(T)
    nm = f"vis_{body}_{rgba_name(rgb)}"
    write_obj(os.path.join(MESHDIR, nm + ".obj"), V, F)
    MESH_ASSETS.append((nm, nm + ".obj")); MATS[rgba_name(rgb)] = rgb
    VIS_GEOMS[body].append((nm, rgba_name(rgb)))
for (body, ln, pn), Ps in HULL.items():
    V, F = hull_mesh(np.concatenate(Ps) - origin(body))
    nm = f"col_{pn}" if f"col_{pn}" not in [x[0] for x in MESH_ASSETS] else f"col_{ln}_{pn}"
    write_obj(os.path.join(MESHDIR, nm + ".obj"), V, F)
    MESH_ASSETS.append((nm, nm + ".obj"))
    COL_GEOMS[body].append(nm)
mesh_bytes = sum(os.path.getsize(os.path.join(MESHDIR, f)) for _, f in MESH_ASSETS)
log(f"meshes: {sum(1 for n, _ in MESH_ASSETS if n.startswith('vis_'))} visual + "
    f"{sum(1 for n, _ in MESH_ASSETS if n.startswith('col_'))} collision hulls, {mesh_bytes / 1e6:.2f} MB")
for b, p in PADS.items():
    log(f"pads {b}: {len(p)} at", [tuple(np.round(x, 1)) for x in p])


# ------------------------------------------------------------------ xml
def f3(v, s=1e-3):
    return " ".join("%.7g" % (x * s) for x in v)


def body_xml(b, ind, free):
    sp = "  " * ind
    par = PARENT.get(b)
    pos = origin(b) - (origin(par) if par else 0)
    if b == "torso":
        out = [f'{sp}<body name="torso" pos="{f3(pos) if not free else "0 0 %.7g" % (LIFT * 1e-3)}" childclass="robot">']
        out.append(f'{sp}  ' + ('<freejoint name="root"/>' if free else '<!-- torso welded to the world -->'))
        out.append(f'{sp}  <camera name="track" mode="trackcom" pos="0.9 -0.9 0.45" xyaxes="0.707 0.707 0 -0.25 0.25 0.935"/>')
    else:
        out = [f'{sp}<body name="{b}" pos="{f3(pos)}">']
        jn = JOINT_OF[b]; j = J[jn]
        lo, hi = RANGE[jn]
        jp = np.array(j["point_mm"], float) - origin(b)
        ax = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[j["axis"]]
        if jn in ACT:
            sv = SERVO[ACT[jn]]
            out.append(f'{sp}  <joint name="{jn}" pos="{f3(jp)}" axis="{ax}" range="{math.radians(lo):.6f} {math.radians(hi):.6f}" '
                       f'damping="{sv["stall"] / sv["speed"]:.4f}" armature="{sv["arm"]}"/>  <!-- {ACT[jn]} -->')
        else:
            out.append(f'{sp}  <joint name="{jn}" class="passive" pos="{f3(jp)}" axis="{ax}" '
                       f'range="{math.radians(lo):.6f} {math.radians(hi):.6f}"/>  <!-- passive, closed loop -->')
    m, c, I = inertial(b)
    merged = [k for k, v in MERGE.items() if v == b]
    out.append(f'{sp}  <inertial mass="{m * 1e-3:.7g}" pos="{f3(c - origin(b))}" '
               f'fullinertia="{I[0, 0] * 1e-9:.7g} {I[1, 1] * 1e-9:.7g} {I[2, 2] * 1e-9:.7g} {I[0, 1] * 1e-9:.7g} {I[0, 2] * 1e-9:.7g} {I[1, 2] * 1e-9:.7g}"/>'
               + (f'  <!-- includes {", ".join(merged)} (parallel axis) -->' if merged else ""))
    for nm, mat in VIS_GEOMS[b]:
        out.append(f'{sp}  <geom class="visual" mesh="{nm}" material="{mat}"/>')
    cls = "col_leg" if b in LEG_BITS else "col"
    for nm in COL_GEOMS[b]:
        out.append(f'{sp}  <geom class="{cls}" name="{nm}" mesh="{nm}"/>')
    for i, pc in enumerate(PADS.get(b, [])):
        out.append(f'{sp}  <geom class="pad" name="{b}_pad{i}" pos="{f3(pc + np.array([0, 0, 3.0]) - origin(b))}"/>')
    if b == "head":
        out.append(f'{sp}  <site name="face" pos="{f3(np.array([60.0, 0, 420.0]) - origin(b))}" size="0.003"/>')
    for ch in BODIES:
        if PARENT.get(ch) == b:
            out += body_xml(ch, ind + 1, free)
    out.append(f"{sp}</body>")
    return out


FLAP_NOTE = ("hand flaps into the arms (welded at 0 deg:\n       in the real robot the flap opens with |arm angle| via a cam, 63 deg max)."
             if "R_hand_flap" in MERGE else "no hand flaps (v21 elbow arm: mitten hand on the forearm).")


def model_xml(free):
    cH = " ".join("%.9g" % x for x in cf_head); cE = " ".join("%.9g" % x for x in cf_ear)
    acts = []
    for jn in J:
        if jn in ACT:
            sv = SERVO[ACT[jn]]; lo, hi = RANGE[jn]
            acts.append(f'    <position name="{jn}" joint="{jn}" kp="{sv["kp"]}" ctrlrange="{math.radians(lo):.6f} {math.radians(hi):.6f}" '
                        f'forcerange="{-sv["stall"]} {sv["stall"]}"/>  <!-- {ACT[jn]} -->')
    mats = "\n".join(f'    <material name="{k}" rgba="{v[0]:.3f} {v[1]:.3f} {v[2]:.3f} 1" specular="0.2"/>' for k, v in MATS.items())
    meshes = "\n".join(f'    <mesh name="{n}" file="{f}"/>' for n, f in MESH_ASSETS)
    excl = "\n".join(f'    <exclude body1="{S}_shank" body2="{S}_foot"/>' for S in "RL")
    return f"""<mujoco model="{TAG}{'' if free else '_fixed'}">
  <!-- generated by sim/export_mjcf.py from code_v17/{r.build5.__name__}.py{(" (KITORU_THETA=" + os.environ["KITORU_THETA"] + ")") if "KITORU_THETA" in os.environ else ""} + {os.path.basename(MASSF)}. Do not edit by hand.
       units m kg rad; all body frames world-aligned; zero pose = design zero pose (origin between the feet on the floor).
       Simplifications: nod_rod merged into torso, ear_rod into head, {FLAP_NOTE} head_nod / ear_axle follow
       nod_crank through <equality joint> 4th-order polynomials (closed_loop_fit.json).
{SERVO_NOTE}  -->
  <compiler angle="radian" inertiafromgeom="false" meshdir="{MESHSUB}" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <size memory="20M"/>
  <visual>
    <global offwidth="1600" offheight="1600"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>

  <default>
    <mesh scale="0.001 0.001 0.001" inertia="shell"/>
    <default class="robot">
      <geom friction="0.8 0.005 0.0001"/>
      <joint damping="0.01"/>
      <default class="passive">
        <joint damping="0.001" armature="0.0001"/>
      </default>
      <default class="visual">
        <geom type="mesh" group="1" contype="0" conaffinity="0" density="0"/>
      </default>
      <default class="col">
        <geom type="mesh" group="3" contype="1" conaffinity="2" density="0" rgba="0.9 0.4 0.2 0.4"/>
      </default>
      <!-- shank / ankle link / foot: extra bit 4 so the two legs can touch each other -->
      <default class="col_leg">
        <geom type="mesh" group="3" contype="5" conaffinity="6" density="0" rgba="0.9 0.2 0.6 0.4"/>
      </default>
      <!-- rubber pads under the sole: cyl r 6 mm, h 6 mm, z 0..6 mm. friction 0.8 [estimate: rubber on hard floor] -->
      <default class="pad">
        <geom type="cylinder" size="0.006 0.003" group="3" contype="5" conaffinity="6" density="0"
              friction="0.8 0.005 0.0001" rgba="0.1 0.1 0.1 1"/>
      </default>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.72 0.74 0.76" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
{mats}
{meshes}
  </asset>

  <worldbody>
    <light name="key" pos="0.6 -0.8 1.6" dir="-0.35 0.5 -0.8" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light name="fill" pos="-0.8 0.6 1.2" dir="0.5 -0.4 -0.7" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor" contype="2" conaffinity="1" friction="0.8 0.005 0.0001"/>
    <camera name="front" pos="1.0 -1.0 0.5" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>
{chr(10).join(body_xml("torso", 2, free))}
  </worldbody>

  <contact>
    <!-- same-leg shank vs foot hulls (not parent-child). sweep17: no real contact inside the joint ranges -->
{excl}
  </contact>

  <equality>
    <!-- head_nod = poly(nod_crank), ear_axle = poly(nod_crank), rad, fitted on beta {NOD[0]:g}..{NOD[1]:g} deg (0.25 deg grid).
         solref time constant 0.002 = 2 x timestep (the stiffest MuJoCo allows without refsafe clamping). -->
    <joint name="loop_head" joint1="head_nod" joint2="nod_crank" polycoef="{cH}" solref="0.002 1" solimp="0.95 0.99 0.001"/>
    <joint name="loop_ear" joint1="ear_axle" joint2="nod_crank" polycoef="{cE}" solref="0.002 1" solimp="0.95 0.99 0.001"/>
  </equality>

  <actuator>
{chr(10).join(acts)}
  </actuator>

  <keyframe>
    <key name="zero" qpos="{('0 0 %.7g 1 0 0 0 ' % (LIFT * 1e-3)) if free else ''}{' '.join(['0'] * (len(BODIES) - 1))}" ctrl="{' '.join(['0'] * len(ACT))}"/>
  </keyframe>
</mujoco>
"""


for free, fn in ((True, f"{TAG}.xml"), (False, f"{TAG}_fixed.xml")):
    xml = model_xml(free)
    if not free:     # fixed version: lift the torso so the feet hang 5 cm above the floor
        xml = xml.replace('<body name="torso" pos="0 0 %.7g"' % (LIFT * 1e-3), '<body name="torso" pos="0 0 %.7g"' % (0.05 + LIFT * 1e-3))
    open(os.path.join(OUT, fn), "w").write(xml)

import mujoco
m = mujoco.MjModel.from_xml_path(os.path.join(OUT, f"{TAG}.xml"))
log(f"compiled {TAG}.xml: nbody {m.nbody} njnt {m.njnt} nq {m.nq} nu {m.nu} ngeom {m.ngeom} neq {m.neq}")
mf = mujoco.MjModel.from_xml_path(os.path.join(OUT, f"{TAG}_fixed.xml"))
log(f"compiled {TAG}_fixed.xml: nq {mf.nq} nu {mf.nu}")
open(os.path.join(OUT, "export_log.txt" if TAG == "kitoru_v17" else f"export_log_{TAG}.txt"), "w").write("\n".join(LOG) + "\n")
