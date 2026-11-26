"""Check horn / idler protrusions and cable exits of every X430 servo against the rest of the model
(zero pose). Geometry from the ROBOTIS XC430 STEP (DC11 dummy asm):
  case x +-14.25, y -35.25..11.25 (horn axis at y 0), z +-17 (horn face +z)
  horn  disc r10.25, z 17..20.9      idler disc r10.25, z -20..-17
  JST plugs are inside the case under the cable cover; cables leave the case
    (a) through the side slots  x = +-14.25 face, y -19.5..-8.7, z -17..-10.9
    (b) through the hollow idler bore (r~4) behind the idler face
"""
import sys, importlib, itertools, numpy as np
import geom4, build4
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build16")
links, joints, _ = mod.build_model()
G = lambda a, b: np.stack(np.meshgrid(*[np.arange(lo + 0.25, hi, 0.5) for lo, hi in zip(a, b)], indexing="ij"), -1).reshape(-1, 3)
def cyl_pts(r, z0, z1):
    P = G((-r, -r, z0), (r, r, z1)); return P[np.hypot(P[:, 0], P[:, 1]) < r]
ZONES = {
    "horn":        cyl_pts(10.25, 17.0, 20.9),
    "idler":       cyl_pts(10.25, -20.0, -17.0),
    "exit_side+x": G((14.5, -19.5, -17.0), (21.25, -8.7, -10.9)),
    "exit_side-x": G((-21.25, -19.5, -17.0), (-14.5, -8.7, -10.9)),
    "exit_bore":   cyl_pts(4.0, -28.0, -20.0),
}
AX = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
SERVOS = []
for ln, L in links.items():
    for pn, ss in L["parts"].items():
        if "servo" in pn.lower() and "horn" not in pn and "idler" not in pn:
            s = ss[0]; ax = s[1][0]; b = build4.aabb(s)
            SERVOS.append((ln, pn, ax, b))
def frame(ax, b):
    """horn axis point + Y_l (toward horn end) from the 11.25 offset; returns (origin, Y_l, axis vector)"""
    k = "xyz".index(ax)
    others = [i for i in range(3) if i != k]
    lens = {i: b[i][1] - b[i][0] for i in others}
    il = max(others, key=lambda i: lens[i])        # 46.5 direction
    iw = min(others, key=lambda i: lens[i])        # 28.5 direction
    best = None
    for j in joints:
        if j["axis"] != ax: continue
        p = np.array(j["point_mm"], float)
        for end, sgn in ((b[il][1], 1), (b[il][0], -1)):
            if abs(abs(end - p[il]) - 11.25) < 0.05 and abs(p[iw] - (b[iw][0] + b[iw][1]) / 2) < 0.05:
                o = p.copy(); o[k] = (b[k][0] + b[k][1]) / 2
                Y = np.zeros(3); Y[il] = sgn
                best = (o, Y, AX[ax], j["name"])
    return best
others = [(ln, pn, s, build4.aabb(build4.sampled(s))) for ln, L in links.items() for pn, ss in L["parts"].items() for s in ss]
def blocked(P, skip):
    hits = {}
    lo, hi = P.min(0), P.max(0)
    for ln, pn, s, bb in others:
        if pn == skip: continue
        base = skip.split("_servo")[0]
        if ("horn" in pn or "idler" in pn) and (base in pn or pn.startswith(base.split("_")[0] + "_" + base.split("_")[1] if "_" in base else base)): continue
        if any(hi[i] < bb[i][0] or lo[i] > bb[i][1] for i in range(3)): continue
        m = geom4.inside(s, P, 0.05)
        if m.any(): hits[f"{ln}:{pn}"] = hits.get(f"{ln}:{pn}", 0) + int(m.sum())
    return hits
out = {}
for ln, pn, ax, b in SERVOS:
    f = frame(ax, b)
    if f is None:
        print(pn, "no joint found"); continue
    o, Y, A, jn = f
    for zs in (1, -1):
        Z = A * zs; X = np.cross(Y, Z)
        R = np.stack([X, Y, Z], 1)
        res = {}
        for zn, P in ZONES.items():
            W = P @ R.T + o
            h = blocked(W, pn)
            frac = sum(h.values()) / len(P)
            res[zn] = (round(frac, 2), sorted(h, key=lambda k: -h[k])[:3])
        out[(pn, zs)] = res
        print(f"{pn:28s} joint {jn:14s} horn {'+' if zs > 0 else '-'}{ax}: " +
              "  ".join(f"{zn} {v[0]:.2f}" for zn, v in res.items()))
        for zn, v in res.items():
            if v[0] > 0.02: print(f"      {zn}: {v[1]}")
