"""v21 elbow arm (build21_arm.py): checks, print parts, mass, ranges, clearances -> out21_arm/.

usage (code_v17/):  ../sim/.venv/bin/python arm21.py [steps...]      steps: parts mass sweep post all (default all)
  parts : regression gate (non-arm printed parts = out20 parts.json), arm print parts STEP + parts.json, arm STEP
  mass  : link mass / CoM / inertia (mass_props 'light') -> mass_build21_arm_light.json, arm21_joints_mass.json
  sweep : collision sweeps (sweep17 point method on surface samples) + distances (KD tree) -> sweep_arm21.json
          elbow self range, shoulder x elbow map vs torso / head (nod -20 / 0 / +28) / legs, shoulder limits vs
          the head over -20 .. +28, walking fold vs hip roll outward, getup keyframes (sim/getup success table)
  post  : torso distances without the shoulder parts, nod range for the raised-arm getup keys (updates the json)
Angles: geom4.rot (+deg about +y turns +x toward -z); shoulder / elbow negative = forward, R hip roll negative = outward.
"""
import os, sys, json, math, time, pickle, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build21_arm as BM
import build4, geom4, printcheck
from geom4 import rot, apply
from build import DISC

OUT = os.path.abspath(BM.OUT)
CACHE = os.environ.get("ARM21_CACHE")
PROCS = int(os.environ.get("PROCS", "16"))
STEPS = sys.argv[1:] or ["all"]


def model():
    if CACHE and os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    m = BM.build_model()
    if CACHE:
        pickle.dump(m, open(CACHE, "wb"))
    return m


t0 = time.time()
links, joints, fasteners = model()
J = {j["name"]: j for j in joints}
print(f"model {time.time() - t0:.0f} s"); sys.stdout.flush()
ARM_PARTS = lambda nm: [f"arm_{nm}_upper_frame", f"arm_{nm}_elbow_bracket_out", f"arm_{nm}_elbow_bracket_in",
                        f"arm_{nm}_hand_frame", f"arm_{nm}_horn_adapter"]
ALL_ARM = set(ARM_PARTS("R") + ARM_PARTS("L"))


# ======================================================================= parts
def up_dir(pn):
    s = "-y" if "_R_" in pn else "+y"
    return {"upper_frame": s, "elbow_bracket_out": s, "elbow_bracket_in": s, "hand_frame": "+z",
            "horn_adapter": "+y"}[next(k for k in BM.FORCE if k in pn)]


# export_parts.py runs a build at import time: its rotation helper is copied here
ROT = {"+z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "-z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
       "+x": [[0, 1, 0], [0, 0, 1], [1, 0, 0]], "-x": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
       "+y": [[0, 0, 1], [1, 0, 0], [0, 1, 0]], "-y": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]}


def rot_solid(s, R, t=(0, 0, 0)):
    from build import IDX, UVI
    R = np.array(R, float); t = np.array(t, float)
    kind, args, rgb = s
    if kind == "loft":
        import loft; return ("loft", loft.rotate(args, R, t), rgb)
    if kind in ("cyl", "scap"):
        return (kind, (tuple(R @ np.array(args[0], float) + t), tuple(R @ np.array(args[1], float))) + tuple(args[2:]), rgb)
    axis, a0, a1, outer, inners, holes = args
    k = IDX[axis]; kn = int(np.argmax(np.abs(R[:, k]))); sgn = R[kn, k]
    naxis = "xyz"[kn]; iu, iv = UVI[axis]; nu, nv = UVI[naxis]

    def mp(u, v):
        p = np.zeros(3); p[iu], p[iv] = u, v
        q = R @ p + t
        return q[nu], q[nv]
    loop = lambda l: [(*mp(u, v), r) for u, v, r in l]
    b0, b1 = sorted((sgn * a0 + t[kn], sgn * a1 + t[kn]))
    return ("rprism", (naxis, b0, b1, loop(outer), [loop(l) for l in inners], [(*mp(u, v), r) for u, v, r in holes]), rgb)


def overhang_enclosed(A, h):
    """printcheck.overhang, except that a patch closed in on every side by supported cells of its own layer (one
    anchor ring) also counts as a bridge when its half-width is <= BRIDGE / 2 (printcheck wants two anchors)"""
    from scipy import ndimage
    st = np.ones((3, 3), bool)
    need = 0.0
    for k in range(1, A.shape[2]):
        cur, below = A[:, :, k], A[:, :, k - 1]
        if not cur.any():
            continue
        sup = ndimage.binary_dilation(below, st)
        un = cur & ~sup
        if not un.any():
            continue
        lab, n = ndimage.label(un, st)
        supported = cur & sup
        dist = ndimage.distance_transform_edt(un) * h
        for i in range(1, n + 1):
            comp = lab == i
            edge = ndimage.binary_dilation(comp, st) & ~comp
            ring = edge & supported
            _, n_anchor = ndimage.label(ring, st)
            reach = ndimage.distance_transform_edt(~ring) * h
            enclosed = bool(edge.any()) and bool((edge == ring).all())
            if reach[comp].max() <= printcheck.CANTILEVER:
                continue
            if (n_anchor >= 2 or enclosed) and dist[comp].max() <= printcheck.BRIDGE / 2:
                continue
            need += comp.sum() * h * h
    return need


def step_parts():
    from scipy import ndimage
    pdir = os.path.join(OUT, "print_parts"); os.makedirs(pdir, exist_ok=True)
    # ---- regression gate: every other printed part has its v20 volume
    ref = {r["part"]: r["volume_cm3"] for r in json.load(open(os.path.join(HERE, "..", "out20", "print_parts", "parts.json")))}
    gate = []
    for ln, pn, sol in printcheck.printed_parts(links):
        if pn in ALL_ARM:
            continue
        if pn in ref:
            v = round(sum(build4.volume(s) for s in sol) / 1000, 1)
            gate.append(dict(part=pn, v20=ref[pn], v21=v, same=abs(v - ref[pn]) < 0.051))
    bad = [g for g in gate if not g["same"]]
    print(f"regression gate: {len(gate)} parts compared, {len(bad)} differ", bad[:5]); sys.stdout.flush()
    assert not bad, bad
    # ---- arm print parts
    table = []
    for ln, pn, sol in printcheck.printed_parts(links):
        if pn not in ALL_ARM:
            continue
        up = up_dir(pn)
        rs = [rot_solid(s, ROT[up]) for s in sol]
        bbs = [build4.aabb(build4.sampled(s)) for s in rs]
        lo = [min(b[i][0] for b in bbs) for i in range(3)]; hi = [max(b[i][1] for b in bbs) for i in range(3)]
        rs = [rot_solid(s, np.eye(3), (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])) for s in rs]
        h = 0.6 if max(b - a for a, b in zip(lo, hi)) < 90 else 1.0
        V, _ = printcheck.voxelize(rs, h)
        A = V[ndimage.find_objects(V.astype(int))[0]]
        need, worst = printcheck.overhang(A, h)
        need_b = overhang_enclosed(A, h)
        build4.write({pn: {"origin": (0, 0, 0), "parts": {pn: rs}}}, os.path.join(pdir, f"{pn}.stp"))
        # support for every orientation (choice table), world voxels; printcheck rule and enclosed roofs as bridges
        Vw, _ = printcheck.voxelize(sol, h)
        Vw = Vw[ndimage.find_objects(Vw.astype(int))[0]]
        sup = {d: (round(printcheck.overhang(printcheck.orient(Vw, d), h)[0]), round(overhang_enclosed(printcheck.orient(Vw, d), h)))
               for d in printcheck.DIRS}
        table.append(dict(part=pn, link=ln, up=up, size_mm="x".join(str(round(b - a)) for a, b in zip(lo, hi)),
                          support_mm2=round(need_b), support_printcheck_mm2=round(need), support_all=sup, volume_cm3=round(sum(build4.volume(s) for s in sol) / 1000, 2),
                          note=next(v for k, v in BM.NOTE.items() if k in pn),
                          colours=sorted({tuple(round(c, 2) for c in s[2]) for s in sol})))
        print(f"{pn:28s} {up:>3s} {table[-1]['size_mm']:>12s} sup {round(need_b):5d} (printcheck {round(need)}) {table[-1]['volume_cm3']:6.2f} cm3", sup); sys.stdout.flush()
    json.dump(table, open(os.path.join(pdir, "parts.json"), "w"), indent=1, ensure_ascii=False)
    # ---- arm assembly STEP (official servo bodies): both arms + the torso-fixed shoulder parts
    keep_t = [k for k in links["torso"]["parts"] if any(t in k for t in ("arm_R_", "arm_L_", "shoulder_shaft_"))]
    sub = {"torso_shoulder_parts": {"origin": (0, 0, 0), "parts": {k: links["torso"]["parts"][k] for k in keep_t}}}
    for nm in "RL":
        for ln in (f"{nm}_arm", f"{nm}_forearm"):
            sub[ln] = links[ln]
    BM.write(sub, os.path.join(OUT, "arms_v21_assembly.stp"))
    return dict(gate=dict(compared=len(gate), differ=len(bad)), parts=table)


# ======================================================================= mass
def mass():
    import mass_props as MP
    old = json.load(open(os.path.join(HERE, "mass_build20_light.json")))
    prof = MP.PROFILES["light"]
    new, parts = {}, []
    for ln in ("R_arm", "R_forearm", "L_arm", "L_forearm", "torso"):
        L = links[ln]; acc = []
        for pn, sol in L["parts"].items():
            r = MP.part_props(pn, sol, prof)
            if r is None:
                continue
            info, W, lo = r
            m, c, I = MP.props_from_vox(W, lo, MP.H)
            info.update(link=ln, part=pn, mass=round(float(m), 1), com=[round(float(x), 1) for x in c]); parts.append(info)
            acc.append((m, c, I))
        M = sum(a[0] for a in acc); C = sum(a[0] * a[1] for a in acc) / M
        I = np.zeros((3, 3))
        for m, c, Ii in acc:
            d = c - C; I += Ii + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        o = np.array(L["origin"], float)
        new[ln] = dict(origin=list(o), mass_g=float(M), com_world=[float(x) for x in C], com_local=[float(x) for x in C - o],
                       inertia_gmm2=I.tolist())
        print(f"{ln:10s} {M:7.1f} g  com {np.round(C, 1)}"); sys.stdout.flush()
    L_all = {k: v for k, v in old["links"].items() if k not in ("R_arm", "L_arm", "R_hand_flap", "L_hand_flap", "torso")}
    L_all.update(new)
    assert set(L_all) == set(links), (set(L_all) ^ set(links))
    tot = sum(v["mass_g"] for v in L_all.values())
    com = sum(np.array(v["com_world"]) * v["mass_g"] for v in L_all.values()) / tot
    keep_parts = [p for p in old["parts"] if p["link"] not in new and p["link"] not in ("R_hand_flap", "L_hand_flap")] + parts
    json.dump(dict(links=L_all, parts=keep_parts, total_g=tot, com=list(com),
                   note="v20 mass_build20_light.json for every link not rebuilt here; R/L_arm, R/L_forearm, torso recomputed (build21_arm, profile light)"),
              open(os.path.join(OUT, "mass_build21_arm_light.json"), "w"), indent=1)
    arm_old = {nm: old["links"][f"{nm}_arm"]["mass_g"] + old["links"][f"{nm}_hand_flap"]["mass_g"] for nm in "RL"}
    arm_new = {nm: new[f"{nm}_arm"]["mass_g"] + new[f"{nm}_forearm"]["mass_g"] for nm in "RL"}
    res = dict(total_v20_g=old["total_g"], total_v21_g=tot, com_v21=list(com), com_v20=old["com"],
               arm_v20_g=arm_old, arm_v21_g=arm_new, torso_v20_g=old["links"]["torso"]["mass_g"], torso_v21_g=new["torso"]["mass_g"],
               parts=[p for p in parts if p["link"] != "torso" or "arm_" in p["part"]])
    # joint table + link mass for sim/export_mjcf.py (SI units like out20/link_mass_v20.csv)
    SPEC = {"XL430-W250": (1.4, 5.45)}
    jt = []
    for j in joints:
        if j["name"].endswith(("arm_swing", "elbow")):
            jt.append(dict(j, stall_Nm_11V=SPEC["XL430-W250"][0], no_load_rad_s_11V=SPEC["XL430-W250"][1]))
    lk = {}
    for ln in ("R_arm", "R_forearm", "L_arm", "L_forearm"):
        v = new[ln]; I = np.array(v["inertia_gmm2"]) * 1e-9
        lk[ln] = dict(parent_joint=next(j["name"] for j in joints if j["child"] == ln), origin_mm=v["origin"], mass_kg=round(v["mass_g"] / 1000, 4),
                      com_from_origin_m=[round(x / 1000, 5) for x in v["com_local"]],
                      inertia_about_com_kgm2=dict(Ixx=I[0, 0], Iyy=I[1, 1], Izz=I[2, 2], Ixy=I[0, 1], Ixz=I[0, 2], Iyz=I[1, 2]))
    res["joints"] = jt; res["links"] = lk
    return res


# ======================================================================= sweep machinery
def solids(ln):
    return [(pn, s) for pn, ss in links[ln]["parts"].items() if "joint_disc" not in pn for s in ss
            if not (s[0] == "cyl" and tuple(s[2]) == tuple(DISC))]


FIX = {ln: solids(ln) for ln in links}
# torso without the shoulder parts that touch the arm servo by design (horn, idler, adapter, shaft): for distances
FIX["torso_body"] = [(pn, s) for pn, s in FIX["torso"] if not (pn.startswith(("arm_R_", "arm_L_")) or pn.startswith("shoulder_shaft"))]
BB = {id(s): np.array(build4.aabb(build4.sampled(s)), float) for ss in FIX.values() for _, s in ss}
_PTS = {}


def surf(s, step):
    G = geom4.samples(s, step)
    if len(G) == 0:
        return G
    keep = np.zeros(len(G), bool)
    for d in np.eye(3) * step:
        for sg in (1, -1):
            keep |= ~geom4.inside(s, G + sg * d)
    return G[keep]


def pts(ln, step):
    k = (ln, step)
    if k not in _PTS:
        _PTS[k] = [(pn, surf(s, step)) for pn, s in FIX[ln]]
        _PTS[k] = [(pn, P) for pn, P in _PTS[k] if len(P)]
    return _PTS[k]


def hits(mv, M, fx, Mf=None, step=1.0, margin=0.05):
    out = []
    Mi = np.linalg.inv(Mf) @ M if Mf is not None else M
    for pn, P in pts(mv, step):
        Q = apply(Mi, P); qlo, qhi = Q.min(0), Q.max(0)
        for fn, f in FIX[fx]:
            b = BB[id(f)]
            if np.any(qhi < b[:, 0]) or np.any(qlo > b[:, 1]):
                continue
            m = np.all((Q >= b[:, 0]) & (Q <= b[:, 1]), axis=1)
            if m.any():
                n = int(geom4.inside(f, Q[m], margin).sum())
                if n:
                    out.append((pn, fn, n))
    return out


def all_pts(ln_M, step):
    """world points of several posed links: {ln: M}"""
    return np.vstack([apply(M, P) for ln, M in ln_M.items() for _, P in pts(ln, step)])


def Rj(name, deg):
    j = J[name]; return rot(j["axis"], deg, j["point_mm"])


import head21 as H21
_A = np.array(H21.NOD_A); _pin0 = H21.S + np.array([H21.NOD_R, 0.0]); _P1, _P2, _LE = H21.ear_geom()


def head_M(beta):
    al = H21.nod(beta)["alpha"]; ps = H21.ear(al)["psi"]
    Mh = Rj("head_nod", beta); Me = Mh @ rot("y", ps, (H21.E0[0], 0, H21.E0[1]))
    return {"head": Mh, "ears": Me}


LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
LEG_L = ["hip_yaw_link", "hip_roll_link", "thigh", "shank", "ankle_link", "foot"]


def leg_M(S, ang):
    out, M = {}, np.eye(4)
    for jn, ln in zip(LEG, LEG_L):
        M = M @ Rj(f"{S}_{jn}", ang.get(jn, 0.0)); out[f"{S}_{ln}"] = M
    return out


def arm_M(nm, a, e):
    Ma = Rj(f"{nm}_arm_swing", a)
    return {f"{nm}_arm": Ma, f"{nm}_forearm": Ma @ Rj(f"{nm}_elbow", e)}


LEGS0 = {**leg_M("R", {}), **leg_M("L", {})}
HEADS = {b: head_M(b) for b in (-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28)}


def env_hits(mv, M, legs=LEGS0, betas=(-20, 0, 28), step=1.0):
    """moving link vs torso, head + ears at the given nods, legs (posed)"""
    h = [("torso",) + x for x in hits(mv, M, "torso", step=step)]
    for b in betas:
        for ln, Mf in HEADS[b].items():
            h += [(f"{ln}@{b}",) + x for x in hits(mv, M, ln, Mf, step=step)]
    for ln, Mf in legs.items():
        h += [(ln,) + x for x in hits(mv, M, ln, Mf, step=step)]
    return h


def _free(ok, keys):
    """contiguous free interval around 0 in sorted keys"""
    ks = sorted(keys); i0 = ks.index(0)
    lo = hi = 0
    for k in reversed(ks[:i0]):
        if not ok[k]:
            break
        lo = k
    for k in ks[i0 + 1:]:
        if not ok[k]:
            break
        hi = k
    return lo, hi


def _pool(fn, items):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(PROCS) as p:
        return p.map(fn, items, chunksize=1)


# ---- workers (module level for the fork pool)
def w_self(args):
    nm, e = args
    M = arm_M(nm, 0, e)
    h = hits(f"{nm}_forearm", M[f"{nm}_forearm"], f"{nm}_arm", M[f"{nm}_arm"], step=0.6)
    return (nm, e, h)


def w_upper(args):
    nm, a = args
    M = arm_M(nm, a, 0)
    return (nm, a, env_hits(f"{nm}_arm", M[f"{nm}_arm"]))


def w_fore(args):
    nm, a, e = args
    M = arm_M(nm, a, e)
    return (nm, a, e, env_hits(f"{nm}_forearm", M[f"{nm}_forearm"]))


def w_shoulder_head(args):
    nm, a, e, b = args
    M = arm_M(nm, a, e)
    h = []
    for ln in (f"{nm}_arm", f"{nm}_forearm"):
        for hl, Mf in HEADS[b].items():
            h += [(hl,) + x for x in hits(ln, M[ln], hl, Mf, step=0.6)]
        if b == 0:
            h += [("torso",) + x for x in hits(ln, M[ln], "torso", step=0.6)]
    return (nm, a, e, b, h)


def mindist(arm, other, step=0.5, step_b=None):
    from scipy.spatial import cKDTree
    A = all_pts(arm, step); T = cKDTree(A)
    d, _ = T.query(all_pts(other, step_b or step), k=1)
    return float(d.min())


def mindist_pair(arm, other, step=0.5):
    """smallest distance and the two parts it is between"""
    from scipy.spatial import cKDTree
    A, la = [], []
    for ln, M in arm.items():
        for pn, P in pts(ln, step):
            A.append(apply(M, P)); la += [pn] * len(P)
    B, lb = [], []
    for ln, M in other.items():
        for pn, P in pts(ln, step):
            B.append(apply(M, P)); lb += [pn] * len(P)
    d, i = cKDTree(np.vstack(A)).query(np.vstack(B), k=1)
    k = int(np.argmin(d))
    return float(d[k]), la[i[k]], lb[k]


def w_fold(args):
    nm, a, e, roll, pitch = args
    S = nm
    legs = leg_M(S, {"hip_roll": roll, "hip_pitch": pitch})
    arm = arm_M(nm, a, e)
    d, pa, pl = mindist_pair(arm, legs)
    h = []
    for ln, M in arm.items():
        for lg, Mf in legs.items():
            h += hits(ln, M, lg, Mf, step=0.6)
    return (nm, a, e, roll, pitch, round(d, 2), h, pa, pl)


KEYS = [  # sim/getup GETUP_STUDY.md, success XL 100+100 prone: (s) shoulder, elbow, hip pitch, knee, ankle pitch
    (1, -33, -130, -61, 0, 22), (2, -85, -58, -72, 1, -25), (3, -110, 46, -115, 4, 22),
    (4, -110, -14, -29, 65, 19), (5, -110, -49, -115, 0, 22), (6, -90, -28, 22, 115, 9)]


def w_key(args):
    k, a, e, hp, kn, ap, b = args
    legs = {**leg_M("R", {"hip_pitch": hp, "knee": kn, "ankle_pitch": ap}), **leg_M("L", {"hip_pitch": hp, "knee": kn, "ankle_pitch": ap})}
    out = {}
    for nm in "RL":
        arm = arm_M(nm, a, e)
        h = []
        for ln, M in arm.items():
            h += env_hits(ln, M, legs=legs, betas=(b,), step=0.6)
        h += hits(f"{nm}_forearm", arm[f"{nm}_forearm"], f"{nm}_arm", arm[f"{nm}_arm"], step=0.6)
        d_head = mindist(arm, HEADS[b], 0.7, 1.0)
        d_torso = mindist(arm, {"torso_body": np.eye(4)}, 0.7, 1.0)
        d_leg = mindist(arm, legs, 0.7, 1.0)
        out[nm] = dict(hits=h, d_head=round(d_head, 1), d_torso=round(d_torso, 1), d_legs=round(d_leg, 1))
    return (k, b, out)


def prewarm(pairs):
    t = time.time()
    for ln, st in pairs:
        pts(ln, st)
    print("prewarm", pairs, round(time.time() - t), "s"); sys.stdout.flush()


def sweep():
    res = {}
    arms = [f"{nm}_{k}" for nm in "RL" for k in ("arm", "forearm")]
    legs = [f"{S}_{l}" for S in "RL" for l in LEG_L]
    prewarm([(ln, st) for ln in arms for st in (1.0, 0.6, 0.5, 0.7)] + [(ln, st) for ln in legs for st in (0.5, 1.0)]
            + [("torso_body", 1.0), ("head", 1.0), ("ears", 1.0)])
    # ---- 1. elbow vs its own upper arm, 1 deg, both arms
    E = list(range(-180, 181))
    r = _pool(w_self, [(nm, e) for nm in "RL" for e in E])
    res["elbow_self"] = {}
    for nm in "RL":
        ok = {e: not h for n, e, h in r if n == nm}
        lo, hi = _free(ok, E)
        first = {str(e): sorted(set((x[0], x[1]) for x in h)) for n, e, h in r if n == nm and e in (lo - 1, hi + 1)}
        res["elbow_self"][nm] = dict(free=(lo, hi), first=first)
        print("elbow self", nm, lo, hi, first); sys.stdout.flush()
    # ---- 5. getup keyframes vs head (-20 / 0 / +28), torso, legs (posed), own upper arm
    items = [(k, a, e, hp, kn, ap, b) for k, a, e, hp, kn, ap in KEYS for b in (-20, 0, 28)]
    rk = _pool(w_key, items)
    res["getup_keys"] = []
    for k, b, out in sorted(rk):
        res["getup_keys"].append(dict(key=k, nod=b, R=out["R"], L=out["L"]))
        print("key", k, "nod", b, {nm: (len(out[nm]["hits"]), out[nm]["d_head"], out[nm]["d_torso"], out[nm]["d_legs"]) for nm in "RL"}); sys.stdout.flush()
    # ---- 6. push-up pose for the figure: key 2 legs + elbow -58, body pitch and shoulder solved so that the hand's
    #      lowest point is right under the shoulder axis and on the floor through the lowest toe point
    res["pushup"] = pushup()
    print("pushup", res["pushup"]); sys.stdout.flush()
    # ---- 2. map: upper arm per shoulder angle, forearm per (shoulder, elbow), 5 deg, step 1.0
    A = list(range(-130, 131, 5)); EE = list(range(-150, 151, 5))
    ru = _pool(w_upper, [(nm, a) for nm in "RL" for a in A])
    rf = _pool(w_fore, [(nm, a, e) for nm in "RL" for a in A for e in EE])
    U = {(n, a): h for n, a, h in ru}; Fo = {(n, a, e): h for n, a, e, h in rf}
    selfok = {nm: res["elbow_self"][nm]["free"] for nm in "RL"}
    res["map"] = {}
    for nm in "RL":
        grid = []
        for a in A:
            row = []
            for e in EE:
                lo, hi = selfok[nm]
                row.append(int(bool(U[(nm, a)]) or bool(Fo[(nm, a, e)]) or not (lo <= e <= hi)))
            grid.append(row)
        sh0 = _free({a: not grid[A.index(a)][EE.index(0)] for a in A}, A)
        el_by_a = {}
        for a in A:
            if U[(nm, a)]:
                continue
            ok = {e: not grid[A.index(a)][EE.index(e)] for e in EE}
            el_by_a[a] = _free(ok, EE) if ok[0] else None
        what = {}
        for (n, a), h in U.items():
            if n == nm and h:
                what[f"upper@{a}"] = sorted(set((x[0], x[2]) for x in h))[:4]
        for (n, a, e), h in Fo.items():
            if n == nm and h and a in (-110, -90, -60, -30, -15, 0, 30, 60, 90, 110) and e in (-150, -135, -130, -120, -90, 90, 120, 130, 135, 150):
                what[f"fore@{a},{e}"] = sorted(set((x[0], x[2]) for x in h))[:4]
        res["map"][nm] = dict(shoulder=A, elbow=EE, blocked=grid, shoulder_free_at_elbow0=sh0,
                              elbow_free_by_shoulder={str(k): v for k, v in el_by_a.items()}, contacts=what)
        print("map", nm, "shoulder free at elbow 0:", sh0, {a: el_by_a.get(a) for a in (-110, -90, -15, 0, 90, 110)}); sys.stdout.flush()
    # ---- 3. shoulder limits vs head over the nod range, 1 deg near the ends (step 0.6), elbow 0 / -90 / +90
    items = [("R", a, e, b) for e in (0, -90, 90) for a in list(range(100, 136)) + list(range(-135, -99)) for b in HEADS]
    rs = _pool(w_shoulder_head, items)
    res["shoulder_head"] = {}
    for e in (0, -90, 90):
        lim = {}
        for b in HEADS:
            okp = {a: not h for n, a, e2, b2, h in rs if e2 == e and b2 == b and a > 0}
            okn = {a: not h for n, a, e2, b2, h in rs if e2 == e and b2 == b and a < 0}
            hi = next((a - 1 for a in sorted(okp) if not okp[a]), 135)
            lo = next((a + 1 for a in sorted(okn, reverse=True) if not okn[a]), -135)
            first = sorted(set((x[0], x[1], x[2]) for n, a, e2, b2, h in rs if e2 == e and b2 == b and a in (hi + 1, lo - 1) for x in h))[:6]
            lim[str(b)] = dict(lo=lo, hi=hi, first=first)
        res["shoulder_head"][str(e)] = lim
        print("shoulder vs head, elbow", e, {b: (v["lo"], v["hi"]) for b, v in lim.items()}); sys.stdout.flush()
    # ---- 4. walking fold vs hip roll outward (R negative, L positive), hip pitch -40 / 0 / +20
    rolls = [0, 8, 12, 20, 25, 33.6]
    fold = [("straight", 0, 0), ("fold_-60", 0, -60), ("fold_-90", 0, -90), ("walk", -15, -90),
            ("walk_s-25", -25, -90), ("walk_e-110", -15, -110), ("walk_s-25_e-110", -25, -110)]
    items = [(nm, a, e, (-r_ if nm == "R" else r_), p) for nm in "RL" for _, a, e in fold for r_ in rolls for p in (-40, 0, 20)]
    rw = _pool(w_fold, items)
    res["fold"] = []
    for nm in "RL":
        for tag, a, e in fold:
            row = dict(arm=nm, pose=tag, shoulder=a, elbow=e, by_roll={})
            for r_ in rolls:
                rr = -r_ if nm == "R" else r_
                ds = [(p, d, h, pa, pl) for n, a2, e2, r2, p, d, h, pa, pl in rw if n == nm and a2 == a and e2 == e and r2 == rr]
                p, d, h, pa, pl = min(ds, key=lambda t: t[1])
                row["by_roll"][str(r_)] = dict(min_mm=d, at_pitch=p, between=(pa, pl), any_hit=any(bool(t[2]) for t in ds))
            res["fold"].append(row)
            print("fold", nm, tag, {k: v["min_mm"] for k, v in row["by_roll"].items()}); sys.stdout.flush()
    # first contact roll (0.5 deg) for straight and walking fold, R and L
    res["roll_first_contact"] = {}
    for nm in "R":              # the left side matched the right in the fold table to 0.2 mm
        for tag, a, e in (fold[0], fold[3], fold[4]):
            rr = np.arange(0, 40.01, 0.5)
            items = [(nm, a, e, (-x if nm == "R" else x), p) for x in rr for p in (-40, 0, 20)]
            rw2 = _pool(w_fold, items)
            first = None
            for x in rr:
                xx = -x if nm == "R" else x
                if any(h for n, a2, e2, r2, p, d, h, pa, pl in rw2 if abs(r2 - xx) < 1e-9):
                    first = float(x); break
            dmin = {str(x): min(d for n, a2, e2, r2, p, d, h, pa, pl in rw2 if abs(r2 - (-x if nm == "R" else x)) < 1e-9) for x in (0, 10, 12, 20, 25, 30, 33.5)}
            res["roll_first_contact"][f"{nm}_{tag}"] = dict(first_contact_roll_out_deg=first, min_mm=dmin)
            print("roll first contact", nm, tag, first, dmin); sys.stdout.flush()
    return res


def pushup(E=-58.0, legs_ang=(("hip_pitch", -72.0), ("knee", 1.0), ("ankle_pitch", -25.0))):
    from scipy.optimize import fsolve
    la = dict(legs_ang)
    legs = {**leg_M("R", la), **leg_M("L", la)}

    def state(phi, a):
        B = rot("y", phi, (0.0, 0.0, 0.0))
        arm = arm_M("R", a, E)
        Ph = all_pts({k: B @ v for k, v in arm.items() if k.endswith("forearm")}, 1.0)
        Pf = all_pts({k: B @ v for k, v in legs.items() if k.endswith("foot")}, 1.0)
        sh = apply(B @ arm["R_arm"], np.array([[0.0, -50.0, BM.SHZ]]))[0]
        return Ph[int(np.argmin(Ph[:, 2]))], Pf[:, 2].min(), sh

    def f(v):
        hp, fz, sh = state(*v)
        return [hp[0] - sh[0], hp[2] - fz]
    phi, a = fsolve(f, [70.0, -60.0])
    hp, fz, sh = state(phi, a)
    out = {}
    for nm in "RL":
        arm = arm_M(nm, a, E)
        h = []
        for ln, M in arm.items():
            h += env_hits(ln, M, legs=legs, betas=(-20, 0, 28), step=0.6)
        out[nm] = dict(hits=h, d_head=round(min(mindist(arm, HEADS[b], 0.7, 1.0) for b in (-20, 0, 28)), 1),
                       d_torso=round(mindist(arm, {"torso_body": np.eye(4)}, 0.7, 1.0), 1), d_legs=round(mindist(arm, legs, 0.7, 1.0), 1))
    return dict(body_pitch_deg=round(float(phi), 2), shoulder_deg=round(float(a), 2), elbow_deg=E, legs=la,
                residual_mm=[round(float(x), 2) for x in f([phi, a])], hand_low=[round(float(x), 1) for x in hp], floor_z=round(float(fz), 1),
                shoulder_axis=[round(float(x), 1) for x in sh], check=out)


def w_post_torso(args):
    kind, k, a, e, hp, kn, ap = args
    legs = {**leg_M("R", {"hip_pitch": hp, "knee": kn, "ankle_pitch": ap}), **leg_M("L", {"hip_pitch": hp, "knee": kn, "ankle_pitch": ap})}
    return (kind, k, {nm: round(mindist(arm_M(nm, a, e), {"torso_body": np.eye(4)}, 0.7, 1.0), 1) for nm in "RL"})


def w_post_nod(args):
    k, a, e, b = args
    h = []
    for nm in "RL":
        for ln, M in arm_M(nm, a, e).items():
            for hl, Mf in HEADS[b].items():
                h += hits(ln, M, hl, Mf, step=0.6)
    return (k, b, len(h))


def post():
    """torso distances without the shoulder parts (the first sweep run measured 0.6 mm to the shoulder horn) and,
    for the keys with the arm raised (shoulder -110), the nod range without arm / head contact (1 deg)"""
    path = os.path.join(OUT, "sweep_arm21.json")
    s = json.load(open(path))
    global HEADS
    HEADS = {b: head_M(b) for b in range(-20, 29)}
    prewarm([(f"{nm}_{k}", st) for nm in "RL" for k in ("arm", "forearm") for st in (0.6, 0.7)] + [("torso_body", 1.0)])
    pu = s["pushup"]; la = pu["legs"]
    items = [("key", k, a, e, hp, kn, ap) for k, a, e, hp, kn, ap in KEYS] + \
            [("pushup", 0, pu["shoulder_deg"], pu["elbow_deg"], la["hip_pitch"], la["knee"], la["ankle_pitch"])]
    for kind, k, d in _pool(w_post_torso, items):
        if kind == "pushup":
            for nm in "RL":
                s["pushup"]["check"][nm]["d_torso"] = d[nm]
        else:
            for row in s["getup_keys"]:
                if row["key"] == k:
                    for nm in "RL":
                        row[nm]["d_torso"] = d[nm]
    raised = [(k, a, e) for k, a, e, hp, kn, ap in KEYS if a <= -100]
    rn = _pool(w_post_nod, [(k, a, e, b) for k, a, e in raised for b in range(-20, 29)])
    s["key_nod_free"] = {}
    for k, a, e in raised:
        bad = sorted(b for kk, b, n in rn if kk == k and n)
        s["key_nod_free"][str(k)] = dict(shoulder=a, elbow=e, first_contact_nod=(bad[0] if bad else None),
                                         free_nod=(-20, (bad[0] - 1) if bad else 28))
        print("key", k, "nod free up to", s["key_nod_free"][str(k)]); sys.stdout.flush()
    s["post"] = "torso distances: torso without the shoulder horn / idler / adapter / shaft; key_nod_free: 1 deg nod scan"
    json.dump(s, open(path, "w"), indent=1, default=str)


# ======================================================================= strength (hand calc, see ARM_ELBOW_v21.md)
def strength():
    T = {"worst": 3.5, "stall": 1.4}                 # elbow N m: getup study (XM runs 2.9-4.4, XL stall 1.4)
    t, b, d = BM.PL1 - BM.PL0, 2 * 13.85, 2 * BM.HOLE_HORN       # horn plate at the lower horn-hole row
    W_plate = t * (b ** 3 - d ** 3) / (6 * b)
    xi, xw = BM.XI, BM.XW; hy = 2 * BM.SLIM; tw = xw - xi
    I_ub = 2 * (hy * tw ** 3 / 12 + hy * tw * ((xi + xw) / 2) ** 2)
    W_ub = I_ub / xw
    F_hand, lat = 16.0, 0.3 * 16.0                   # N: half of 3.3 kg body, lateral friction 0.3
    W_weak = 2 * (b * t ** 2 / 6)
    H_, B_ = 2 * BM.XW, 2 * BM.PL1
    W_hand = (B_ * H_ ** 3 - (B_ - 2 * BM.TW) * (H_ - 2 * BM.TW) ** 3) / 12 / BM.XW
    return dict(
        elbow_torque_Nm=T,
        horn_plate=dict(section="horn plate at the lower horn holes, width 27.7, thickness 3.5, printed flat (in-layer)",
                        W_mm3=round(W_plate), stress_MPa={k: round(v * 1000 / W_plate, 1) for k, v in T.items()},
                        both_plates_MPa={k: round(v * 1000 / W_plate / 2, 1) for k, v in T.items()}),
        upper_frame_elbow_bay=dict(section="two x walls 3.85 x 41.2 at x 14.6..18.45 (servo case as web), printed -y/+y (in-layer)",
                                   W_mm3=round(W_ub), stress_MPa={k: round(v * 1000 / W_ub, 2) for k, v in T.items()}),
        horn_screw_bearing=dict(note="4 x M2 on r 8, all torque in bearing (no clamp friction): F = T / (4 * 8 mm), sigma = F / (2 mm * 3.5 mm)",
                                force_N={k: round(v * 1000 / 32, 1) for k, v in T.items()},
                                stress_MPa={k: round(v * 1000 / 32 / 7.0, 1) for k, v in T.items()}),
        plates_lateral=dict(note=f"side push {lat:.1f} N at the hand (110 mm), two plates bending out of plane, printed flat (in-layer)",
                            W_mm3=round(W_weak), stress_MPa=round(lat * 110 / W_weak, 1)),
        hand_section=dict(note="hand shell at z 110 (printed +z, across layers): moment = 16 N x 24 mm, hollow 36.9 x 49.4, wall 2",
                          stress_MPa=round(F_hand * 24 / W_hand, 2)),
        reference="v20 leg: interlayer repeat limit ~7 MPa, target 5 MPa [estimate] (LEG_PRINT_KIT_R.md 6)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    do = lambda k: "all" in STEPS or k in STEPS
    if do("parts"):
        res = step_parts()
        json.dump(res["gate"], open(os.path.join(OUT, "regression_gate.json"), "w"), indent=1)
    if do("mass"):
        m = mass()
        m["strength"] = strength()
        json.dump(m, open(os.path.join(OUT, "arm21_joints_mass.json"), "w"), indent=1)
    if do("sweep"):
        t1 = time.time()
        s = sweep()
        s.update(model="build21_arm", surface_samples_mm=(1.0, 0.6), margin_mm=0.05, seconds=round(time.time() - t1))
        json.dump(s, open(os.path.join(OUT, "sweep_arm21.json"), "w"), indent=1, default=str)
    if do("post"):
        post()
    print("done", round(time.time() - t0), "s")
