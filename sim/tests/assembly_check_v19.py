"""v19 assembly check with the real X430 geometry (official STEP sampled by code_v17/x430_geom.py).

(A) every servo: real servo surface vs all other solids at the zero pose: interference points, minimum gap
(B) right leg + right hip yaw: cable exit slots and connector access (free space outside the side slots)
(C) cable route between consecutive servos (voxel search inside the links' envelope) + joint slack
(D) straight insertion of each servo into its own link (6 directions, 60 mm)
(E) screwdriver access (cylinder d 6 x 60 mm on the screw axis) for every case screw and horn screw of the
    right leg / right yaw servo; parts that are assembled later (shells, covers) are listed separately
Run: cd sim && uv run python tests/assembly_check_v19.py   -> out19/ASSEMBLY_CHECK_v19.md, out19/assembly_check/*
"""
import json, math, os, sys, collections, heapq
import numpy as np
from scipy.spatial import cKDTree
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(SIM, "..", "code_v17")
sys.path.insert(0, CODE)
import build19 as B, build4, geom4, x430_geom as XG, x430_holes as X
OUT = os.path.join(SIM, "..", "out19"); FIG = os.path.join(OUT, "assembly_check"); os.makedirs(FIG, exist_ok=True)

links, joints, _ = B.build_model()
J = {j["name"]: j for j in joints}
SKIP, frames = B.servo_step_parts(links)
P_s, G_s, GN = XG.points()
SOL = [(ln, pn, i, s, np.array(build4.aabb(build4.sampled(s)), float)) for ln, L in links.items() for pn, ss in L["parts"].items()
       if pn not in SKIP for i, s in enumerate(ss)]
LATER = ("shell", "cover", "shorts", "lid", "hatch", "thigh_shell", "shank_shell")
RIGHT = ["R_ankle_roll_servo_XM430", "R_ankle_pitch_servo_XC430", "R_knee_servo_XM430", "R_hip_pitch_servo_XM430", "yaw_servo_R_XC430"]
REPORT = {}
SECTIONS = os.environ.get("SECTIONS", "ABCDE")
_cache = os.path.join(FIG, "assembly_check_v19.json")
if os.path.exists(_cache) and SECTIONS != "ABCDE":
    REPORT = json.load(open(_cache))
BOX = [(fr["link"], sv, 0, links[fr["link"]]["parts"][sv][0], np.array(build4.aabb(build4.sampled(links[fr["link"]]["parts"][sv][0])), float))
       for sv, fr in frames.items()]


def Rm(fr):
    return np.c_[fr["X"], fr["Y"], fr["Z"]]


def occupied(Q, exclude=(), only_links=None, pad=0.0, margin=0.0, boxes=False, only_parts=None):
    """bool per point: inside any solid (optionally a subset; boxes=True adds the servo case boxes)"""
    Q = np.asarray(Q, float); m = np.zeros(len(Q), bool); who = collections.Counter()
    lo, hi = Q.min(0) - pad, Q.max(0) + pad
    for ln, pn, i, s, bb in SOL + (BOX if boxes else []):
        if only_parts is not None and not any(t in pn for t in only_parts):
            continue
        if (ln, pn) in exclude or pn in exclude or (only_links is not None and ln not in only_links):
            continue
        if np.any(bb[:, 1] < lo) or np.any(bb[:, 0] > hi):
            continue
        sel = np.all((Q >= bb[:, 0] - pad) & (Q <= bb[:, 1] + pad), axis=1)
        if not sel.any():
            continue
        ins = geom4.inside(s, Q[sel], margin)
        if ins.any():
            idx = np.nonzero(sel)[0][ins]; m[idx] = True; who[pn] += int(ins.sum())
    return m, who


# ---------------- (A) real servo vs printed parts
A = REPORT.get("A_real_servo", {})
for sv, fr in (frames.items() if "A" in SECTIONS else []):
    keep = np.isin(G_s, [GN.index(g) for g in ("case", "horn", "idler", "cable_cover")])
    Pw = P_s[keep] @ Rm(fr).T + fr["O"]; Gw = G_s[keep]
    m, who = occupied(Pw, margin=0.02)
    # minimum gap: volume samples of nearby solids
    lo, hi = Pw.min(0) - 4, Pw.max(0) + 4
    near = [(ln, pn, s) for ln, pn, i, s, bb in SOL if not (np.any(bb[:, 1] < lo) or np.any(bb[:, 0] > hi))]
    best = {}
    for ln, pn, s in near:
        Q = geom4.samples(s, 0.4)
        if not len(Q):
            continue
        Q = Q[np.all((Q >= lo) & (Q <= hi), axis=1)]
        if not len(Q):
            continue
        d, _ = cKDTree(Q).query(Pw, k=1, distance_upper_bound=4.0)
        for g in np.unique(Gw):
            dg = d[Gw == g]
            if np.isfinite(dg).any():
                v = float(dg.min())
                if GN[g] not in best or v < best[GN[g]][0]:
                    best[GN[g]] = (v, pn)
    A[sv] = dict(link=fr["link"], interfering_points=int(m.sum()), interfering_parts=dict(who.most_common(6)),
                 min_gap_mm={g: (round(v[0], 2), v[1]) for g, v in best.items()})
    print("A", sv, A[sv], flush=True)
REPORT["A_real_servo"] = A

# ---------------- (B) cable exit slots + connector access (side slots of the idler half, both sides)
SLOT = dict(y=(-19.5, -8.7), z=(-17.0, -10.9))
PLUG = dict(y=(-18.85, -9.35), z=(-13.3, -6.2))


def side_box(fr, sg, d0, d1, win, step=0.7):
    ys = np.arange(win["y"][0], win["y"][1] + 1e-9, step); zs = np.arange(win["z"][0], win["z"][1] + 1e-9, step)
    ds = np.arange(d0, d1 + 1e-9, step)
    Yy, Zz, Dd = np.meshgrid(ys, zs, ds, indexing="ij")
    Pq = np.c_[(14.25 + Dd.ravel()) * sg, Yy.ravel(), Zz.ravel()]
    return Pq @ Rm(fr).T + fr["O"]


Bc = REPORT.get("B_connectors", {})
for sv in (RIGHT if "B" in SECTIONS else []):
    fr = frames[sv]; Bc[sv] = {}
    for sg in (1, -1):
        m3, w3 = occupied(side_box(fr, sg, 0.3, 3.0, SLOT))
        m15, w15 = occupied(side_box(fr, sg, 0.3, 15.0, PLUG))
        Bc[sv]["+x" if sg > 0 else "-x"] = dict(exit_open=round(1 - m3.mean(), 2), exit_blockers=dict(w3.most_common(3)),
                                                  plug_access_open=round(1 - m15.mean(), 2), access_blockers=dict(w15.most_common(4)))
    print("B", sv, Bc[sv], flush=True)
REPORT["B_connectors"] = Bc


# ---------------- (C) cable routes (voxel A*), endpoints = outside the open side slot (2 mm)
def exit_point(sv):
    fr = frames[sv]
    side = max(("+x", "-x"), key=lambda k: (Bc.get(sv, {}).get(k, {}).get("exit_open", 0)))
    sg = 1 if side == "+x" else -1
    p = np.array([sg * (14.25 + 2.5), np.mean(SLOT["y"]), np.mean(SLOT["z"])])
    return Rm(fr) @ p + fr["O"], side


OC = [s for ln, pn, i, s, bb in SOL if pn == "opencr_ttl_plugs_cables"]
oc_bb = np.array(build4.aabb(build4.sampled(OC[0])), float) if OC else None


def route(pa, pb, link_set, h=1.0, dil=1):
    lo = np.minimum(pa, pb) - 45; hi = np.maximum(pa, pb) + 45
    ax = [np.arange(lo[i], hi[i] + h, h) for i in range(3)]
    shape = tuple(len(a) for a in ax)
    Gp = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    occ, _ = occupied(Gp, boxes=True); occ = occ.reshape(shape)
    # envelope: inside the convex hull of the solids of the links the cable crosses
    from scipy.spatial import Delaunay
    pts = np.vstack([geom4.samples(s, 3.0) for ln, pn, i, s, bb in SOL if ln in link_set and not (np.any(bb[:, 1] < lo) or np.any(bb[:, 0] > hi))])
    env = Delaunay(pts).find_simplex(Gp) >= 0
    env = env.reshape(shape)
    from scipy import ndimage
    blocked = ndimage.binary_dilation(occ, iterations=dil) | ~env
    ia = tuple(np.clip(np.round((pa - lo) / h).astype(int), 0, np.array(shape) - 1))
    ib = tuple(np.clip(np.round((pb - lo) / h).astype(int), 0, np.array(shape) - 1))
    for p in (ia, ib):                    # endpoints sit in the slot: free a small ball around them
        sl = tuple(slice(max(0, p[k] - 3), p[k] + 4) for k in range(3))
        blocked[sl] = occ[sl] & False
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    dist = {ia: 0.0}; prev = {}; pq = [(0.0, ia)]
    while pq:
        dcur, u = heapq.heappop(pq)
        if u == ib:
            break
        if dcur > dist.get(u, 1e18):
            continue
        for d in nb:
            v = (u[0] + d[0], u[1] + d[1], u[2] + d[2])
            if min(v) < 0 or v[0] >= shape[0] or v[1] >= shape[1] or v[2] >= shape[2] or blocked[v]:
                continue
            nd = dcur + h * math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd + h * math.dist(v, ib), v))
    if ib not in dist:
        reach = np.array([k for k in dist])
        far = reach[np.argmin(np.linalg.norm(reach - np.array(ib), axis=1))] if len(reach) else np.array(ia)
        return None, lo + far * h
    path = [ib]
    while path[-1] != ia:
        path.append(prev[path[-1]])
    return dist[ib], np.array([lo + np.array(p) * h for p in path[::-1]])


CHAIN = [("R_ankle_roll_servo_XM430", "R_ankle_pitch_servo_XC430", "R_ankle_pitch", {"R_ankle_link", "R_shank", "R_foot"}),
         ("R_ankle_pitch_servo_XC430", "R_knee_servo_XM430", "R_knee", {"R_shank", "R_thigh"}),
         ("R_knee_servo_XM430", "R_hip_pitch_servo_XM430", "R_hip_pitch", {"R_thigh", "R_hip_yaw_link"}),
         ("R_hip_pitch_servo_XM430", "yaw_servo_R_XC430", "R_hip_yaw", {"R_hip_yaw_link", "torso"}),
         ("yaw_servo_R_XC430", "OpenCR", None, {"torso"})]
CHILD_OF = {j["name"]: j["child"] for j in joints}


def link_pose(ln, ang):
    """design pose 4x4 of a leg link for joint angles (deg)"""
    chain = []
    cur = ln
    while cur != "torso":
        jn = next(j for j in joints if j["child"] == cur); chain.append(jn); cur = jn["parent"]
    M = np.eye(4)
    for jn in chain[::-1]:
        M = M @ geom4.rot(jn["axis"], ang.get(jn["name"], 0.0), jn["point_mm"])
    return M


Cc = REPORT.get("C_cable_routes", [])
if "C" in SECTIONS:
    Cc = []
for a, b, jn, lset in (CHAIN if "C" in SECTIONS else []):
    pa, sa = exit_point(a)
    if b == "OpenCR":
        pb = oc_bb.mean(1) if oc_bb is not None else None; sb = "TTL"
    else:
        pb, sb = exit_point(b)
    L_, path = route(pa, pb, lset)
    slack = None
    if jn:
        lo_, hi_ = J[jn]["lower_deg"], J[jn]["upper_deg"]
        la, lb = frames[a]["link"], frames[b]["link"]
        ds = []
        for t in np.linspace(lo_, hi_, 25):
            Ma, Mb = link_pose(la, {jn: t}) if la != "torso" else np.eye(4), link_pose(lb, {jn: t}) if lb != "torso" else np.eye(4)
            ds.append(np.linalg.norm((Ma[:3, :3] @ pa + Ma[:3, 3]) - (Mb[:3, :3] @ pb + Mb[:3, 3])))
        slack = dict(joint=jn, range_deg=[lo_, hi_], end_distance_mm=[round(min(ds), 1), round(max(ds), 1)],
                     zero_mm=round(float(np.linalg.norm(pa - pb)), 1), extra_length_needed_mm=round(max(ds) - float(np.linalg.norm(pa - pb)), 1))
    rec = dict(frm=a, to=b, exits=[sa, sb], path_found=L_ is not None, path_length_mm=None if L_ is None else round(L_, 1), joint_slack=slack)
    if L_ is None:
        rec["stuck_near_mm"] = [round(float(v), 1) for v in path]
    else:
        np.save(os.path.join(FIG, f"route_{a}_{b}.npy"), path)
    Cc.append(rec); print("C", rec, flush=True)
REPORT["C_cable_routes"] = Cc

# ---------------- (D) straight insertion into the own link
Dd = REPORT.get("D_insertion", {})
STRUCT = ("frame", "bracket", "spacer", "sole", "yaw_bearing", "pelvis")
for sv in (RIGHT if "D" in SECTIONS else []):
    fr = frames[sv]
    keep = np.isin(G_s, [GN.index(g) for g in ("case", "horn", "idler", "cable_cover")])
    Pw = P_s[keep][::3] @ Rm(fr).T + fr["O"]
    res = {}
    for nm, v in (("+x", fr["X"]), ("-x", -fr["X"]), ("+y", fr["Y"]), ("-y", -fr["Y"]), ("+z", fr["Z"]), ("-z", -fr["Z"])):
        free = 60.0; who = {}
        for t in np.arange(1.0, 60.01, 1.0):
            m, w = occupied(Pw + t * v, only_links={fr["link"]}, margin=0.05, only_parts=STRUCT)
            if m.any():
                free = t - 1.0; who = dict(w.most_common(2)); break
        res[nm] = dict(world_dir=[int(round(x)) for x in v], free_mm=free, first_hit=who)
    Dd[sv] = res; print("D", sv, {k: (v["free_mm"], v["world_dir"]) for k, v in res.items()}, flush=True)
REPORT["D_insertion"] = Dd

# ---------------- (E) screwdriver access
log = json.load(open(os.path.join(OUT, "servo_seat_log.json")))
screws = []; seen = set()
for h in log["case_holes"]:
    if h["servo"] not in RIGHT or (h["servo"], h["hole"]) in seen:
        continue
    seen.add((h["servo"], h["hole"]))
    fr = frames[h["servo"]]; p, n = X.HOLES[h["hole"] - 1]
    pw, nw = X.to_world(fr, p, n)
    screws.append(("case M2.5", h["servo"], h["hole"], pw, nw))
HPOS = [(5.657, 5.657), (-5.657, 5.657), (5.657, -5.657), (-5.657, -5.657)]
for sv in RIGHT:
    fr = frames[sv]
    for k, (x, y) in enumerate(HPOS):
        pw = fr["O"] + Rm(fr) @ np.array([x, y, 20.9]); screws.append(("horn M2", sv, k + 1, pw, fr["Z"]))


def driver(pw, nw, r=3.0, length=60.0, rim=2.0):
    """screw head side: march on a line rim mm off the axis (the hole itself is empty) through the gap and the
    plate(s); the driver cylinder starts where the material ends"""
    a = nw / np.linalg.norm(nw); ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, ref); u /= np.linalg.norm(u); v = np.cross(a, u)
    d, inmat, last = 0.05, False, 0.0
    while d < 30:
        m, _ = occupied((pw + rim * u + d * a)[None])
        if m[0]:
            inmat = True; last = d
        elif inmat and d - last > 0.6:
            break
        d += 0.2
    d = last if inmat else 0.0
    start = pw + (d + 0.3) * a
    pts = [start + t * a + rr * (math.cos(th) * u + math.sin(th) * v) for t in np.arange(0, length, 1.5)
           for rr in (0.0, 1.5, 3.0) for th in (np.linspace(0, 2 * math.pi, 12, endpoint=False) if rr > 0 else [0.0])]
    return np.array(pts), d


Ee = REPORT.get("E_driver", [])
if "E" in SECTIONS:
    Ee = []
for kind, sv, k, pw, nw in (screws if "E" in SECTIONS else []):
    Q, depth = driver(pw, nw)
    m, who = occupied(Q)
    own_leg = sv[0] if sv[0] in "RL" else ("R" if "_R_" in sv else "L")
    def other_assembly(p):
        ln = next((l for l, pn, i, s_, bb in SOL if pn == p), "")
        return (ln.startswith(("L_", "R_")) and not ln.startswith(own_leg + "_")) or "arm" in ln or ln in ("head", "ears")
    later = {p: c for p, c in who.items() if any(t in p for t in LATER) or other_assembly(p)}
    hard = {p: c for p, c in who.items() if p not in later}
    Ee.append(dict(kind=kind, servo=sv, n=k, axis_world=[int(round(x)) for x in nw], material_mm=round(depth, 2),
                   clear=not hard, blocked_by=hard, later_parts=later))
    print("E", Ee[-1], flush=True)
REPORT["E_driver"] = Ee
json.dump(REPORT, open(os.path.join(FIG, "assembly_check_v19.json"), "w"), indent=1, default=str)
print("saved")
