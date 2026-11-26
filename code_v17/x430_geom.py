"""Real X430 servo geometry from the official ROBOTIS STEP (references/XC-430_idle.stp) without a CAD kernel.

gmsh / OCC could not be loaded on this machine (libGLU.so.1 missing), so the B-rep is sampled directly:
  * PLANE faces: boundary loops (LINE, CIRCLE arcs, B-spline control polygons) projected to the plane, interior
    filled with a grid (even-odd rule) -> surface points
  * CYLINDRICAL / CONICAL faces: (angle, height) range of the boundary, filled on that rectangle (holes inside a
    curved face are ignored: slightly conservative)
  * TOROIDAL / SPHERICAL / B-spline faces (fillets, small): boundary samples only
Every face is tagged with its product (case, horn, idler, cable cover, header, plug, screws ...).
Coordinates = servo frame of the handover (horn axis origin, horn +z, long axis +y) - the STEP is already in it.

usage: python3 x430_geom.py   -> writes x430_points.npz next to this file and prints a summary
"""
import math, os, re, collections
import numpy as np
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = os.path.join(HERE, "..", "references", "XC-430_idle.stp")
CACHE = os.path.join(HERE, "x430_points.npz")
GROUP = {"DC11_B01_CASE_M_DUMMY": "case", "DC11_B01_CASE_F_DUMMY": "case", "DC11_B01_CASE_B_DUMMY2": "case",
         "M_DC11_A01_HORN": "horn", "DC11_A01_HORN_INSERT_1": "horn", "M_DC11_A01_IDLER": "idler",
         "M_DC11_A01_IDLER_CAP": "idler", "DC11_A01_IDLER_SCREW": "idler", "DC14_A01_INSERT_1": "idler",
         "DC11_A01_CVR_CABLE": "cable_cover", "B3B-EH": "header", "EHR-3": "plug", "PHS_M2_5X12_K": "screw",
         "FHS_M2_5X06_NYLOK": "screw", "DX124_SCREW-COVER": "case", "DC11_A01_WIN_LED": "case"}


def load(step=STEP, spacing=0.5):
    t = open(step).read()
    data = t.split("DATA;")[1].split("ENDSEC;")[0]
    D = {int(m.group(1)): " ".join(m.group(2).split()) for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|$)", data, re.S)}
    refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", re.sub(r"'[^']*'", "", s))]
    nums = lambda s: [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]

    def vec(i):
        return np.array(nums(D[i].split("(", 1)[1].split("(", 1)[1].split(")")[0]))

    def ax2(i):
        r = refs(D[i]); o = vec(r[0])
        z = vec(r[1]) if len(r) > 1 else np.array([0, 0, 1.0]); x = vec(r[2]) if len(r) > 2 else np.array([1.0, 0, 0])
        z = z / np.linalg.norm(z); x = x - x.dot(z) * z
        if np.linalg.norm(x) < 1e-9:
            x = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0]); x = x - x.dot(z) * z
        x /= np.linalg.norm(x)
        return o, x, np.cross(z, x), z

    # ---- product names, reps, transforms (same scheme as references/parse.py)
    pname = {i: re.match(r"PRODUCT\(\s*'([^']*)'", v).group(1) for i, v in D.items() if v.startswith("PRODUCT(")}
    form = {i: refs(v)[0] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_FORMATION")}
    pdef = {i: pname[form[refs(v)[0]]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION(")}
    pds = {i: pdef[refs(v)[0]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_SHAPE") and refs(v)[0] in pdef}
    rep2name = {}
    for i, v in D.items():
        if v.startswith("SHAPE_DEFINITION_REPRESENTATION"):
            a, b = refs(v)[:2]
            if a in pds:
                rep2name[b] = pds[a]
    srr = [(refs(v)[0], refs(v)[1]) for v in D.values() if v.startswith("SHAPE_REPRESENTATION_RELATIONSHIP")]
    for a, b in srr:
        if a in rep2name and b not in rep2name:
            rep2name[b] = rep2name[a]
        elif b in rep2name and a not in rep2name:
            rep2name[a] = rep2name[b]

    def M4(i):
        o, x, y, z = ax2(i); M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = o; return M
    parent = {}
    for i, v in D.items():
        if "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION" in v:
            r = refs(v); a1, a2 = refs(D[r[2]])[:2]
            parent.setdefault(r[0], []).append((r[1], M4(a2) @ np.linalg.inv(M4(a1))))

    def world(rep):
        if rep not in parent:
            return [np.eye(4)]
        return [W @ T for p, T in parent[rep] for W in world(p)]
    reps = {i for i, v in D.items() if "SHAPE_REPRESENTATION" in v and not v.startswith(("SHAPE_DEFINITION", "CONTEXT")) and "RELATIONSHIP" not in v}

    # ---- edge sampling
    def edge_pts(e, step=0.4):
        v = D[e]; r = refs(v); p0, p1 = vec(refs(D[r[0]])[0]), vec(refs(D[r[1]])[0]); cv = r[2]
        same = v.rstrip(")").rstrip().endswith(".T.")
        c = D[cv]
        if c.startswith("LINE"):
            n = max(2, int(np.linalg.norm(p1 - p0) / step) + 1)
            return np.linspace(p0, p1, n)
        if c.startswith("CIRCLE"):
            o, x, y, z = ax2(refs(c)[0]); R = nums(c.split(",")[-1])[0]
            a0 = math.atan2((p0 - o) @ y, (p0 - o) @ x); a1 = math.atan2((p1 - o) @ y, (p1 - o) @ x)
            if np.linalg.norm(p0 - p1) < 1e-7:
                a1 = a0 + (2 * math.pi if same else -2 * math.pi)
            elif same:
                while a1 <= a0: a1 += 2 * math.pi
            else:
                while a1 >= a0: a1 -= 2 * math.pi
            n = max(3, int(abs(a1 - a0) * R / step) + 1)
            a = np.linspace(a0, a1, n)
            return o + R * (np.cos(a)[:, None] * x + np.sin(a)[:, None] * y)
        # B-spline or other: control points / referenced points
        pts = [vec(j) for j in refs(c) if D[j].startswith("CARTESIAN_POINT")]
        if not pts:
            for j in refs(c):
                pts += [vec(k) for k in refs(D[j]) if D[k].startswith("CARTESIAN_POINT")]
        return np.array([p0] + pts + [p1])

    def face_loops(f):
        loops = []
        for b in refs(D[f]):
            if not D[b].startswith(("FACE_OUTER_BOUND", "FACE_BOUND")):
                continue
            lp = refs(D[b])[0]
            pts = []
            for oe in refs(D[lp]):
                ec = refs(D[oe])[-1]
                q = edge_pts(ec)
                if D[oe].rstrip(")").rstrip().endswith(".F."):
                    q = q[::-1]
                pts.append(q)
            loops.append(np.concatenate(pts))
        return loops

    def sample_face(f):
        r = refs(D[f]); surf = [j for j in r if not D[j].startswith(("FACE_OUTER_BOUND", "FACE_BOUND"))][0]
        S = D[surf]; loops = face_loops(f)
        if not loops:
            return np.zeros((0, 3))
        B = np.concatenate(loops)
        if S.startswith("PLANE"):
            o, x, y, z = ax2(refs(S)[0])
            uv = [np.c_[(L - o) @ x, (L - o) @ y] for L in loops]
            U = np.concatenate(uv); lo, hi = U.min(0), U.max(0)
            g = [np.arange(lo[i] + spacing / 2, hi[i], spacing) for i in range(2)]
            if min(len(g[0]), len(g[1])) == 0:
                return B
            G = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 2)
            inside = np.zeros(len(G), bool)
            for L in uv:
                inside ^= Path(L).contains_points(G)
            P = o + G[inside, 0:1] * x + G[inside, 1:2] * y
            return np.vstack([B, P])
        if S.startswith(("CYLINDRICAL_SURFACE", "CONICAL_SURFACE")):
            o, x, y, z = ax2(refs(S)[0]); vals = nums(S.split(",", 2)[-1]); R0 = vals[0]
            ang = math.radians(vals[1]) if S.startswith("CONICAL") else 0.0
            d = B - o; h = d @ z; th = np.arctan2(d @ y, d @ x)
            ts = np.sort(th); gaps = np.diff(np.r_[ts, ts[0] + 2 * math.pi]); k = int(np.argmax(gaps))
            if gaps[k] < math.radians(25):
                t0, t1 = -math.pi, math.pi
            else:
                t0 = ts[(k + 1) % len(ts)]; t1 = t0 + 2 * math.pi - gaps[k]
            h0, h1 = h.min(), h.max()
            Rm = R0 + max(abs(h0), abs(h1)) * math.tan(ang)
            nt = max(3, int((t1 - t0) * max(Rm, 0.5) / spacing)); nh = max(2, int((h1 - h0) / spacing) + 1)
            T, Hh = np.meshgrid(np.linspace(t0, t1, nt), np.linspace(h0, h1, nh), indexing="ij")
            Rr = R0 + Hh * math.tan(ang)
            P = o + (Rr * np.cos(T))[..., None] * x + (Rr * np.sin(T))[..., None] * y + Hh[..., None] * z
            return np.vstack([B, P.reshape(-1, 3)])
        return B

    def faces_of(i, acc, seen):
        if i in seen:
            return
        seen.add(i); v = D[i]
        if v.startswith("ADVANCED_FACE"):
            acc.append(i); return
        if v.startswith(("CARTESIAN_POINT", "DIRECTION", "AXIS2_PLACEMENT", "VERTEX_POINT", "EDGE_LOOP")):
            return
        for j in refs(v):
            if j not in reps:
                faces_of(j, acc, seen)
    out_p, out_g, out_n = [], [], []
    names = sorted(set(GROUP.values()) | {"other"})
    for r in reps:
        fl = []
        for it in refs(D[r])[:-1]:
            if D[it].startswith(("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")):
                faces_of(it, fl, set())
        if not fl:
            continue
        nm = rep2name.get(r, "?"); grp = GROUP.get(nm, "other")
        P = np.vstack([sample_face(f) for f in fl])
        cand = [r] + [a for a, b in srr if b == r] + [b for a, b in srr if a == r]
        pr = next((c for c in cand if c in parent), r)
        for W in world(pr):
            Q = P @ W[:3, :3].T + W[:3, 3]
            out_p.append(Q); out_g.append(np.full(len(Q), names.index(grp)))
    P = np.vstack(out_p); G = np.concatenate(out_g)
    return P, G, names


def points(refresh=False):
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["P"], z["G"], list(z["names"])
    P, G, names = load()
    np.savez_compressed(CACHE, P=P, G=G, names=np.array(names))
    return P, G, names


if __name__ == "__main__":
    P, G, names = points(refresh=True)
    for i, n in enumerate(names):
        m = G == i
        if m.any():
            print(f"{n:12s} {m.sum():7d} pts  min {np.round(P[m].min(0), 2)}  max {np.round(P[m].max(0), 2)}")
