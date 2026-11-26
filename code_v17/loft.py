"""'loft' solid: a band swept along an axis through a list of levels, each level an outer and an inner outline in
the plane normal to the axis, outlines with equal vertex counts, straight (ruled) between levels.

    ("loft", (axis, levels, closed), rgb)
        axis    "x" | "y" | "z"
        levels  ((h, outer[(u, v)...], inner[(u, v)...]), ...) sorted by h
        closed  True: outer/inner are closed loops (a ring band)
                False: outer/inner are open polylines from end A to end B (a C-shaped band, ends are planar)

Side faces are planar when consecutive levels are parallel offsets sampled at the same normal angles
(rounded rectangles via rr_offset); otherwise quads are split into triangles.
"""
import math
import numpy as np
from matplotlib.path import Path

IDX = {"x": 0, "y": 1, "z": 2}
UVI = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


# ---------------------------------------------------------------- outline helpers
def rr_points(cu, cv, A, B, R, n_arc=12, start=None):
    """Rounded rectangle centred (cu, cv), half sizes A, B, corner radius R, CCW from the +u side middle.
    Corner arcs sampled at fixed angles -> offsets of the same shape have parallel edges."""
    R = max(R, 0.0)
    pts = []
    corners = [((cu + A - R, cv + B - R), 0), ((cu - A + R, cv + B - R), 90), ((cu - A + R, cv - B + R), 180),
               ((cu + A - R, cv - B + R), 270)]
    for (ox, oy), a0 in corners:
        for i in range(n_arc + 1):
            t = math.radians(a0 + 90.0 * i / n_arc)
            pts.append((ox + R * math.cos(t), oy + R * math.sin(t)))
    return pts


def clip_open(pts, umin=None, umax=None):
    """Cut a closed CCW loop by u >= umin (or u <= umax) and return the open polyline that stays, starting at the
    first cut point (CCW order). Exactly two crossings are expected."""
    n = len(pts)
    keep = lambda p: (umin is None or p[0] >= umin) and (umax is None or p[0] <= umax)
    cut = umin if umin is not None else umax
    # rotate so that we start at a kept point that follows a dropped one
    start = next(i for i in range(n) if keep(pts[i]) and not keep(pts[i - 1]))
    out = []
    a, b = pts[start - 1], pts[start]
    t = (cut - a[0]) / (b[0] - a[0])
    out.append((cut, a[1] + t * (b[1] - a[1])))
    i = start
    while keep(pts[i % n]):
        out.append(pts[i % n])
        i += 1
    a, b = pts[(i - 1) % n], pts[i % n]
    t = (cut - a[0]) / (b[0] - a[0])
    out.append((cut, a[1] + t * (b[1] - a[1])))
    return out


# ---------------------------------------------------------------- geometry
def region(outer, inner, closed):
    if closed:
        return outer, inner
    return list(outer) + list(inner)[::-1], None


def interp(l0, l1, t):
    return [(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for a, b in zip(l0, l1)]


def area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return s / 2


def volume(args):
    axis, levels, closed = args
    def A(lv):
        o, i = region(lv[1], lv[2], closed)
        return abs(area(o)) - (abs(area(i)) if i is not None else 0)
    return sum((A(levels[k]) + A(levels[k + 1])) / 2 * (levels[k + 1][0] - levels[k][0]) for k in range(len(levels) - 1))


def aabb(args):
    axis, levels, closed = args
    k, (iu, iv) = IDX[axis], UVI[axis]
    us = [p[0] for lv in levels for l in lv[1:] for p in l]
    vs = [p[1] for lv in levels for l in lv[1:] for p in l]
    hs = [lv[0] for lv in levels]
    out = [None, None, None]
    out[k] = (min(hs), max(hs)); out[iu] = (min(us), max(us)); out[iv] = (min(vs), max(vs))
    return out


def inside(args, P, margin=0.0):
    axis, levels, closed = args
    k, (iu, iv) = IDX[axis], UVI[axis]
    P = np.asarray(P, float)
    h = P[:, k]
    uv = P[:, [iu, iv]]
    m = np.zeros(len(P), bool)
    hs = [lv[0] for lv in levels]
    sel = (h > hs[0] + margin) & (h < hs[-1] - margin)
    if not sel.any():
        return m
    idx = np.nonzero(sel)[0]
    hr = np.round(h[idx], 3)
    for hv in np.unique(hr):
        j = idx[hr == hv]
        b = min(max(np.searchsorted(hs, hv) - 1, 0), len(levels) - 2)
        h0, h1 = hs[b], hs[b + 1]
        t = 0.0 if h1 == h0 else (hv - h0) / (h1 - h0)
        o = interp(levels[b][1], levels[b + 1][1], t)
        i = interp(levels[b][2], levels[b + 1][2], t)
        reg, hole = region(o, i, closed)
        r = margin
        pin = Path(np.asarray(reg)).contains_points(uv[j], radius=r) & Path(np.asarray(reg)).contains_points(uv[j], radius=-r)
        if hole is not None:
            hp = Path(np.asarray(hole))
            pin &= ~(hp.contains_points(uv[j], radius=r) | hp.contains_points(uv[j], radius=-r))
        m[j] = pin
    return m


def to_local(args, o):
    axis, levels, closed = args
    k, (iu, iv) = IDX[axis], UVI[axis]
    sh = lambda l: [(u - o[iu], v - o[iv]) for u, v in l]
    return (axis, tuple((h - o[k], sh(a), sh(b)) for h, a, b in levels), closed)


def rotate(args, R, t):
    axis, levels, closed = args
    R = np.asarray(R, float); t = np.asarray(t, float)
    k = IDX[axis]
    kn = int(np.argmax(np.abs(R[:, k])))
    sgn = R[kn, k]
    naxis = "xyz"[kn]
    iu, iv = UVI[axis]
    nu, nv = UVI[naxis]
    def mp(u, v):
        p = np.zeros(3); p[iu], p[iv] = u, v
        q = R @ p + t
        return (float(q[nu]), float(q[nv]))
    lv = [(float(sgn * h + t[kn]), [mp(*p) for p in a], [mp(*p) for p in b]) for h, a, b in levels]
    lv.sort(key=lambda x: x[0])
    return (naxis, tuple(lv), closed)


def P3(axis, u, v, h):
    k, (iu, iv) = IDX[axis], UVI[axis]
    p = [0.0, 0.0, 0.0]; p[k], p[iu], p[iv] = h, u, v
    return tuple(p)


def mesh(args):
    """Vertices (world xyz) and faces (vertex index lists), faces oriented outward, planar (quads split if not)."""
    axis, levels, closed = args
    n = len(levels[0][1])
    V = []
    def vid(p):
        V.append(p); return len(V) - 1
    O = [[vid(P3(axis, u, v, h)) for u, v in a] for h, a, b in levels]
    I = [[vid(P3(axis, u, v, h)) for u, v in b] for h, a, b in levels]
    F = []
    segs = range(n) if closed else range(n - 1)
    nx = lambda s: (s + 1) % n
    for L in range(len(levels) - 1):
        for s in segs:
            F.append([O[L][s], O[L][nx(s)], O[L + 1][nx(s)], O[L + 1][s]])
            F.append([I[L][s], I[L + 1][s], I[L + 1][nx(s)], I[L][nx(s)]])
        if not closed:
            for e in (0, n - 1):
                F.append([O[L][e], I[L][e], I[L + 1][e], O[L + 1][e]])
    for L in (0, len(levels) - 1):
        for s in segs:
            F.append([O[L][s], O[L][nx(s)], I[L][nx(s)], I[L][s]])
    Vn = np.array(V, float)
    # weld coincident vertices (zero-radius corners, clipped ends) so neighbouring faces share edges
    canon, key2id = [], {}
    for p in Vn:
        k = tuple(np.round(p, 6))
        if k not in key2id:
            key2id[k] = len(key2id)
        canon.append(key2id[k])
    Vw = np.zeros((len(key2id), 3))
    for i, c in enumerate(canon):
        Vw[c] = Vn[i]
    Vn = Vw
    F = [[canon[i] for i in f] for f in F]
    out = []
    for f in F:
        # drop repeated points
        keep = [f[0]]
        for q in f[1:]:
            if q != keep[-1]:
                keep.append(q)
        if keep[0] == keep[-1] and len(keep) > 1:
            keep.pop()
        if len(keep) < 3:
            continue
        P = Vn[keep]
        nrm = np.zeros(3)
        for a in range(len(P)):
            nrm += np.cross(P[a], P[(a + 1) % len(P)])
        if np.linalg.norm(nrm) < 1e-12:
            continue
        nrm /= np.linalg.norm(nrm)
        planar = np.max(np.abs((P - P[0]) @ nrm)) < 1e-6
        polys = [keep] if planar else [[keep[0], keep[1], keep[2]], [keep[0], keep[2], keep[3]]]
        for poly in polys:
            Q = Vn[poly]
            nn = np.zeros(3)
            for a in range(len(Q)):
                nn += np.cross(Q[a], Q[(a + 1) % len(Q)])
            if np.linalg.norm(nn) < 1e-12:
                continue
            nn /= np.linalg.norm(nn)
            c = Q.mean(0)
            eps = 0.02
            if inside(args, (c + eps * nn)[None, :])[0] and not inside(args, (c - eps * nn)[None, :])[0]:
                poly = poly[::-1]
            out.append(poly)
    return Vn, out
